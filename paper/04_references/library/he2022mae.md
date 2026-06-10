---
phase: 4
agent: excerpt-curator-1
directives: [T4]
last_modified: 2026-06-11
key: he2022mae
verification_status: VERIFIED_A
verified_by_A: 2026-06-11
corrected_fields: [doi_added, pages_added, mae_patchify_excerpt_resolved]
card_grade: FULL
excerpt_access: abstract_only
---
# Masked Autoencoders Are Scalable Vision Learners

**경고: 이 card의 verbatim 발췌·abstract는 검증/문체 대조 전용 — 논문 본문으로 복사·근접 의역 절대 금지 (A2)**

## 서지 (scout 예비 — 검증 전)
저자: Kaiming He, Xinlei Chen, Saining Xie, Yanghao Li, Piotr Dollár, Ross Girshick
Venue: CVPR 2022, pp.15979–15988
연도: 2022
arXiv: 2111.06377 (submitted November 2021)
publisher DOI: 10.1109/CVPR52688.2022.01553
공식 URL: https://arxiv.org/abs/2111.06377
[A1 추가] pages 15979–15988, DOI 10.1109/CVPR52688.2022.01553 (DBLP + Crossref 확인)

## Abstract 전문 (verbatim)
"This paper shows that masked autoencoders (MAE) are scalable self-supervised learners for computer vision. Our MAE approach is simple: we mask random patches of the input image and reconstruct the missing pixels. It is based on two core designs. First, we develop an asymmetric encoder-decoder architecture, with an encoder that operates only on the visible subset of patches (without mask tokens), along with a lightweight decoder that reconstructs the original image from the latent representation and mask tokens. Second, we find that masking a high proportion of the input image, e.g., 75%, yields a nontrivial and meaningful self-supervisory task. Coupling these two designs enables us to train large models efficiently and effectively: we accelerate training (by 3x or more) and improve accuracy. Our scalable approach allows for learning high-capacity models that generalize well: e.g., a vanilla ViT-Huge model achieves the best accuracy (87.8%) among methods that use only ImageNet-1K data. Transfer performance in downstream tasks outperforms supervised pre-training and shows promising scaling behavior."

## 핵심 발췌 (verbatim, 섹션/위치 표기)

> "we mask random patches of the input image and reconstruct the missing pixels." (Abstract)

커버 claim: C-026, C-033, C-084
활용 맥락: §2.3 및 §3.3에서 우리의 patch masking + 재구성 설계의 직접 계보를 밝힐 때. "본 논문의 patch/masking은 He et al. (CVPR 2022)에서 착안" 1문장 인용.

---

> "we develop an asymmetric encoder-decoder architecture, with an encoder that operates only on the visible subset of patches (without mask tokens), along with a lightweight decoder that reconstructs the original image from the latent representation and mask tokens." (Abstract)

커버 claim: C-026
활용 맥락: 비대칭 인코더-디코더 구조의 원류를 언급할 때. 우리 논문이 이 구조를 시계열에 적용·확장함을 설명.

---

> "masking a high proportion of the input image, e.g., 75%, yields a nontrivial and meaningful self-supervisory task." (Abstract)

커버 claim: C-026, C-033
활용 맥락: 높은 masking ratio가 유의미한 self-supervised 과제를 만든다는 원리를 우리의 patch masking 정당화에 활용. 단, 우리는 이미지가 아닌 시계열이므로 "이미지 맥락" 한정 명시 필요.

**[A1 EXCERPT_RESOLVED — arXiv PDF §3에서 직접 발췌 확보 (2026-06-11):]**

> "our encoder embeds patches by a linear projection with added positional embeddings, and then processes the resulting set via a series of Transformer blocks." (§3, MAE encoder 단락)

커버 claim: C-026, C-033 (linear patchify 설명, §3 MAE encoder 단락에서 확보)

---

## 우리 논문에서의 활용

커버 claim: C-026, C-033, C-084

- **§2.3 Related Work**: C-026 — Vision MAE를 patch masking의 직접 원류로 소개. "우리의 patchify·masking 설계는 He et al. (CVPR 2022) MAE에서 착안" 1문장
- **§3.3 Methodology (Patchify)**: C-033 — linear patchify 사용 근거 ("학습 효율·구현 단순성 + MAE 원류 계보")
- **Appendix §B.1**: C-084 — linear patchify vs patch_cnn 선택 정당화

## 주의사항
- MAE는 이미지(ViT) 대상 — "시계열에서도 MAE가 최적" 식의 과도한 일반화 금지. 우리는 MAE의 masking 원리를 시계열에 적용했음을 서술하는 수준으로 제한.
- arXiv HTML 버전(404) 접근 불가로 본문 발췌 미확보 — 발췌는 abstract에서만 추출. verifier가 CVPR 공식본 또는 arXiv PDF에서 §3(Method) 발췌 보강 필요.
- publisher DOI verifier-TODO 상태 유지.
