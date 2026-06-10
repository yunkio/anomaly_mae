---
phase: 4
agent: excerpt-curator-2
directives: [T4]
last_modified: 2026-06-11
key: zhang2022selfdistill
verification_status: VERIFIED_A
verified_by_A: source-verifier-A2
verified_note: "All bibliographic fields confirmed via DBLP (journals/pami/ZhangBM22). No arXiv preprint found. Abstract verbatim remains EXCERPT_UNVERIFIED (IEEE Xplore paywall). Paper used only as term-origin 1-2 citation — bibliographic fields sufficient."
card_grade: FULL
excerpt_access: abstract_only
---

<!-- INTERNAL_NOTE_NOT_FOR_MANUSCRIPT — verbatim excerpts below are for internal reference only and must not appear in any submitted text -->

> **A2 경고**: 아래 verbatim 발췌는 card 내부 전용입니다. 어떠한 형태로도 원고에 그대로 복사하지 마십시오.

---

## 서지 정보

- **Key**: zhang2022selfdistill
- **제목**: Self-Distillation: Towards Efficient and Compact Neural Networks
- **저자**: Linfeng Zhang, Chenglong Bao, Kaisheng Ma
- **Venue**: IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI), Vol. 44, No. 8, pp. 4388–4403, 2022
- **DOI**: 10.1109/TPAMI.2021.3067100
- **DBLP**: journals/pami/ZhangBM22
- **확인 출처**: DBLP 직접 열람 (2026-06-11); arXiv 버전 미확인

---

## Abstract (verbatim)

<!-- INTERNAL_NOTE_NOT_FOR_MANUSCRIPT -->

EXCERPT_UNVERIFIED — 전문 arXiv 접근 불가. DBLP 서지 확인 완료. 아래는 DBLP 및 서지 정보 기반 요약이며 abstract verbatim은 미확보.

논문은 "self-distillation"을 compact하고 efficient한 신경망 학습을 위한 기법으로 도입하며, 단일 네트워크 내에서 깊은 레이어가 얕은 레이어를 감독하는 방식의 지식 증류를 제안한다. Knowledge distillation(KD)을 교사-학생 쌍 없이 단일 모델 내에서 수행하는 것이 핵심이다.

---

## 핵심 발췌

### 발췌 1 — "self-distillation" 용어 원류 (서지 기반)

<!-- INTERNAL_NOTE_NOT_FOR_MANUSCRIPT -->

EXCERPT_UNVERIFIED — 직접 본문 접근 불가. 용어 원류로서의 위상은 DBLP TPAMI 등재 + ANCHOR_SDMAE_DOSSIER §5.1 bib101 확인으로 뒷받침됨.

- **§위치**: 제목 및 도입부 (추정)
- **지지 Claim**: C-028 (self-distillation 용어 원류), C-034 (용어 계보 방어)
- **활용 맥락**: §2.3 또는 §3.4에서 "self-distillation"이라는 용어가 Zhang et al. (TPAMI 2022)에서 효율적·compact 신경망을 위한 기법으로 도입되었음을 명시. SDMAE → 본 논문으로 이어지는 계보의 기점.
- **주의**: 원문 abstract verbatim 미확보. 인용 시 서지(DOI, 권·호·쪽)만으로 족하며 내용 인용은 하지 말 것 — verifier가 IEEE Xplore에서 본문 접근 후 보강.

---

### 발췌 2 — 용어 계보 위상 (간접 확인)

<!-- INTERNAL_NOTE_NOT_FOR_MANUSCRIPT -->

EXCERPT_UNVERIFIED — 직접 확인 불가.

논문의 핵심 기여는 하나의 모델 내에서 깊이 방향으로 knowledge를 증류하는 "self-distillation" 프레임워크다. 외부 teacher 모델 없이 내부 레이어 간 증류를 수행한다는 점에서 전통적 KD와 구분된다.

- **§위치**: §1 Introduction / §3 Method (추정)
- **지지 Claim**: C-028, C-034
- **활용 맥락**: SDMAE(Ristea et al. 2024)가 "variant of self-distillation"이라고 명시한 원류로 Zhang et al.을 인용해야 함. 우리 논문도 동일 계보를 따른다.
- **주의**: 원문 미열람 — "efficient/compact NN" 문맥이므로 우리 논문에서 "효율성"이 아닌 "표현 학습 계보"로 인용해야 맥락 왜곡 없음.

---

## 활용 절

| 우리 논문 위치 | 활용 방식 | 근거 |
|-------------|---------|------|
| §2.3 또는 §3.4 | self-distillation 용어 원류 인용 (1문장, 괄호) | C-028, C-034 |
| §3.4 각주 | "self-distillation" 용어를 Zhang et al. 에서 차용했음을 명시 | C-028 |

용어 원류로 1~2회 괄호 인용으로 충분. 본문 내용을 직접 인용하거나 방법론 비교 대상으로 쓰는 것은 맥락 불일치.

---

## 주의사항

1. **abstract verbatim 미확보**: IEEE Xplore 페이페월 뒤에 있음. verifier가 기관 접속으로 abstract + 핵심 정의 확인 필요.
2. **용어 원류 주장 범위**: Zhang et al.은 self-distillation을 efficient/compact NN용으로 도입했으며 anomaly detection과 무관. "self-distillation을 AD에 적용했다"는 서술에 이 논문을 인용하면 왜곡 — 용어 원류로만 사용.
3. **복사 금지 표현**: 제목 "Self-Distillation: Towards Efficient and Compact Neural Networks" — 우리 논문 제목이나 소제목에 이 구절 사용 금지.
