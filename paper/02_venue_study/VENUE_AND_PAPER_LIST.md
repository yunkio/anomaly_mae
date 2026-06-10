---
phase: 2
agent: venue-scout
directives: [T2]
last_modified: 2026-06-11
revision: r2 (fixer — adversarial review paper/99_reviews/p2_venue_corpus_r1.md 반영: V-001/V-002 MAJOR + V-003/V-004 MINOR + V-005 NOTE; fixlog: p2_fixlog_r2.md; 정정 이력은 말미 부록)
---

> **경고**: 아래 verbatim 인용문(따옴표 처리된 것)은 분석 전용 — 논문 본문으로 복사 금지 (A2)
> **서지 단서**: 모든 서지 정보는 "Phase 4 공식 소스 재검증 필요". 직접 접근·대조 확인 전까지 arXiv 버전 기준임.

---

# VENUE AND PAPER LIST — Phase 2 탑티어 AI 학회 & 논문 분석

## 1. 탑티어 AI 학회 리스트 (2024–2026 기준)

### 1.1 머신러닝·AI 핵심 학회

| 학회 | 전체 명칭 | 주기 | 특성 | 시계열/이상탐지 강세 |
|-----|---------|------|------|-----------------|
| **NeurIPS** | Conference on Neural Information Processing Systems | 연 1회 (12월) | 이론+실험 균형, oral/spotlight/poster 3단계 | 중상 (벤치마크·평가 논문 강세) |
| **ICML** | International Conference on Machine Learning | 연 1회 (7월) | 이론 강세, 대규모 | 중 |
| **ICLR** | International Conference on Learning Representations | 연 1회 (5월) | 딥러닝·표현학습 특화, OpenReview 완전공개 | 높음 (시계열 모델 수상 다수) |
| **AAAI** | AAAI Conference on Artificial Intelligence | 연 1회 (2월) | 응용 AI 강세 | 중 |
| **IJCAI** | International Joint Conference on AI | 연 1회 (8월) | 전통 AI+ML 혼합 | 중 |
| **UAI** | Conference on Uncertainty in Artificial Intelligence | 연 1회 | 확률 모델 강세 | 낮음 |
| **AISTATS** | Artificial Intelligence and Statistics | 연 1회 (5월) | 통계+ML 교차 | 낮음 |

### 1.2 데이터마이닝·정보 시스템 학회

| 학회 | 전체 명칭 | 주기 | 특성 | 시계열/이상탐지 강세 |
|-----|---------|------|------|-----------------|
| **KDD** | ACM SIGKDD International Conference on Knowledge Discovery and Data Mining | 연 1회 (8월) | 응용 데이터마이닝 최강, research+applied track | **최강** (시계열·이상탐지 핵심 venue) |
| **WWW / TheWebConf** | The Web Conference | 연 1회 (5월) | 웹·소셜·스트림 데이터 강세 | 높음 |
| **VLDB** | International Conference on Very Large Data Bases / PVLDB | 연 1회 | DB+스트리밍 데이터 강세 | 중상 (평가 지표 논문 강세) |
| **ICDM** | IEEE International Conference on Data Mining | 연 1회 (12월) | 전통적 데이터마이닝 | 중 |
| **CIKM** | ACM Conference on Information and Knowledge Management | 연 1회 (10월) | 검색·KG·시계열 혼합 | 중 |
| **SDM** | SIAM International Conference on Data Mining | 연 1회 (4월) | 수치 방법 강세 | 낮음-중 |

### 1.3 시계열·이상탐지 강세 특화 venue

| venue | 특성 |
|-------|------|
| **PVLDB (Proc. VLDB Endow.)** | VUS, Affiliation, PA%K 등 평가지표 논문의 1차 출판처 (Paparrizos et al. 2022, Boniol et al. 2022) |
| **IEEE TKDE** | 트랜잭션, 시계열+이상탐지 응용 다수 |
| **ACM SIGMOD** | DB 출신 스트림 이상탐지 |
| **ECML-PKDD** | 유럽 기준 ML+이상탐지 혼합 |

---

## 2. Elsevier 상위 저널 구조 관례 (별도 절)

타깃 포맷이 Elsevier elsarticle이므로 아래 저널들의 구조 관례를 별도 정리한다.

### 2.1 주요 Elsevier 저널 (AI·ML 분야)

| 저널 | Impact Factor 근사 | 특성 | 논문 길이 |
|-----|------------------|------|---------|
| **Pattern Recognition** | ~8.5 | 컴퓨터비전+패턴인식, 실험 중심 | 10–15p (2단 조판) |
| **Neural Networks** | ~7.8 | 딥러닝 이론+응용, 수식 밀도 높음 | 10–15p |
| **Knowledge-Based Systems** | ~8.8 | 응용 AI, 이상탐지 논문 다수 게재 | 10–15p |
| **Information Sciences** | ~8.1 | 넓은 커버리지, 데이터마이닝 포함 | 10–15p |
| **Expert Systems with Applications** | ~8.5 | 응용 강세, 시계열 센서 이상탐지 다수 | 12–18p |
| **Neurocomputing** | ~6.0 | 뉴럴 아키텍처 구현 중심 | 10–15p |

### 2.2 Elsevier 저널 논문의 구조 관례

직접 접근 가능한 DTAAD (Knowledge-Based Systems, 2024)와 elsarticle 템플릿 기반 관찰:

**전형적 섹션 순서** (9–15페이지 기준):
1. Abstract (구조화: Background / Method / Results / Conclusion 또는 자유형식, 150–250 words)
2. Keywords (5–8개)
3. Introduction (1.5–2.5p): 동기 → 기존 한계 → 기여 → 논문 구성 개요
4. Related Work (1.5–2p): 주제별 소절 2–4개
5. Methodology / Proposed Method (2–4p): 문제 정의 → 전체 구조 개요 → 컴포넌트 상세
6. Experiments (2–4p): 데이터셋 → 구현 세부 → 비교 결과 → Ablation → 분석
7. Conclusion (0.5–1p): 요약 + 한계 + 미래 연구
8. Acknowledgments
9. References

**특징적 차이점 (학회 vs 저널)**:
- 저널은 실험 섹션이 더 길고 상세함 (데이터셋 설명 + 추가 분석 + 계산 비용 + 파라미터 민감도 필수에 가까움)
- Related Work가 독립 섹션으로 분리 (학회 논문은 Introduction에 통합하는 경우도 있음)
- Background/Preliminaries 섹션을 Method 앞에 별도로 두는 경우가 많음
- 수식 번호 매기기 연속적
- elsarticle 두 가지 레이아웃: `preprint` (단열, A4) / `5p` (2단, 저널 최종)
- Highlights 요소 (3–5 bullet, 125 chars/bullet 이하) 요구하는 저널 다수
- Graphical Abstract (선택): 하나의 요약 figure

---

## 3. 고평가 논문 선정 목록 (12–20편)

### 3.1 선정 기준
- 2023–2026 연도 내 출판 또는 accepted (단, 구조 분석 목적상 ICLR 2022 Anomaly Transformer 포함)
- 수상/oral·spotlight/고인용
- **시계열 이상탐지 논문 다수 포함** (필수 요건)
- 본 TSMAE 논문과 직접 연관: MAE/masking, teacher-student, semi-supervised/PU, 이상탐지 평가지표

---

### Paper 1: Anomaly Transformer

| 항목 | 내용 |
|-----|------|
| 제목 | Anomaly Transformer: Time Series Anomaly Detection with Association Discrepancy |
| 저자 | Jiehui Xu, Haixu Wu, Jianmin Wang, Mingsheng Long |
| Venue | **ICLR 2022 (Spotlight)** |
| arXiv | 2110.02642 |
| OpenReview | openreview.net/forum?id=LzQQ89U1qm_ |
| 선정 사유 | ICLR 2022 spotlight, TSAD의 기준 baseline, 벤치마크 총 6개(SWaT/SMD/PSM/SMAP/MSL + NeurIPS-TS 합성 — abstract "six unsupervised time series anomaly detection benchmarks" 재확인, fixer r2 V-005; 이 중 5개가 본 연구와 공유), 관련도 최고 |
| 검증 상태 | 직접 PDF 확인 완료 (2026-06-11) |

**섹션 구조**: Introduction(2p) → Related Work[§2.1 Unsupervised TSAD / §2.2 Transformers for TS](1p) → Method[§3.1 Anomaly Transformer / §3.2 Minimax Association Learning](2p) → Experiments[§4.1 Main Results / §4.2 Model Analysis](2.5p) → Conclusion(0.5p) + Appendix A–D(5p)

**총 페이지**: 9p 본문 + 부록; 참고문헌 포함 12p

**Figures**: Fig.1 아키텍처(Anomaly-Attention 다이어그램), Fig.2 Minimax 최적화 개념도, Fig.3 ROC 곡선 5개 데이터셋, Fig.4 NeurIPS-TS 바 차트, Fig.5 anomaly score 시각화(5 anomaly type × 3 모델), Fig.6 σ 파라미터 시각화, Fig.7 파라미터 민감도(window size, λ), Fig.8 NeurIPS-TS 기준선 비교, Fig.9 실제 데이터셋 anomaly score

**Tables**: Table 1 메인 성능비교(5 datasets, P/R/F1, 18 baselines), Table 2 ablation(anomaly criterion/prior-association/optimization strategy), Table 3 adjacent association weights

---

### Paper 2: DCdetector

| 항목 | 내용 |
|-----|------|
| 제목 | DCdetector: Dual Attention Contrastive Representation Learning for Time Series Anomaly Detection |
| 저자 | Yiyuan Yang, Chaoli Zhang, Tian Zhou, Qingsong Wen, Liang Sun |
| Venue | **KDD 2023** (ACM SIGKDD, Long Beach, CA, August 6–10, 2023) |
| DOI | 10.1145/3580305.3599295 |
| arXiv | 2306.10347 |
| 선정 사유 | KDD 2023 TSAD SOTA, 동일 벤치마크(SWaT/SMD/PSM/SMAP/MSL), 26 baseline 비교, 본 논문 baseline 중 하나 |
| 검증 상태 | 직접 PDF 확인 완료 (2026-06-11) |

**섹션 구조**: Introduction(2p) → Related Work[Time Series Anomaly Detection / Contrastive Representation Learning](1p) → Methodology[§3.1 Overall Architecture / §3.2 Dual Attention Contrastive Structure / §3.3 Representation Discrepancy / §3.4 Anomaly Criterion](3p) → Experiments[§4.1 Benchmark Datasets / §4.2 Baselines & Evaluation / §4.3 Implementation / §4.4 Main Results / §4.5 Model Analysis](5p) → Conclusion(0.5p)

**총 페이지**: 17p (ACM 2단 조판)

**Figures**: Fig.1 3-way 아키텍처 비교(재구성/Anomaly Transformer/DCdetector), Fig.2 DCdetector 전체 아키텍처 flow, Fig.3 채널독립 패칭 개념도, Fig.4 업샘플링 예시, Fig.5 anomaly score 시각화 비교(5 anomaly type, AnomalyTrans vs DCdetector), Fig.6 파라미터 민감도(5가지 하이퍼파라미터), Fig.7 GPU 메모리·시간 비용

**Tables**: Table 1 메인 성능비교(5 datasets, P/R/F1, 26 baselines), Table 2 멀티지표(Aff-P/R, R_A_R, V_ROC 등), Table 3 NIPS-TS 결과, Table 4 UCR 결과, Table 5 NIPS-TS 멀티지표, Table 6 Ablation(stop gradient), Table 7 Ablation(forward process)

---

### Paper 3: SDMAE (Self-Distilled Masked Auto-Encoders)

| 항목 | 내용 |
|-----|------|
| 제목 | Self-Distilled Masked Auto-Encoders are Efficient Video Anomaly Detectors |
| 저자 | Nicolae-Catalin Ristea, Florinel-Alin Croitoru, Radu Tudor Ionescu, Marius Popescu, Fahad Shahbaz Khan, Mubarak Shah |
| Venue | **CVPR 2024** |
| arXiv | 2306.12041 |
| 공식 페이지 | cvpr.thecvf.com/virtual/2024/poster/30615 |
| 선정 사유 | R9/R21 직결 anchor 논문. MAE+self-distillation+teacher-student 구조. 동일 용어 선례. |
| 검증 상태 | 직접 HTML 확인 완료 (상세: ANCHOR_SDMAE_DOSSIER.md) |

**섹션 구조**: Introduction → Related Work[Frame/Cube-Level / Object-Centric / MAE & Knowledge Distillation] → Method[4개 컴포넌트: motion weighting / architecture / self-distillation 2-stage / synthetic anomaly aug] → Experiments[§4.1 Setup / §4.2 Results] → Conclusion + Supplementary[§6.1 / §6.2]

**Figures**: Fig.1 아키텍처, Fig.2 성능-속도 scatter, Fig.3 합성 이상 예시, Fig.4 예측 시각화, Fig.5 재구성 및 이상 맵

**Tables**: Table 1 4 벤치마크 비교(AUC+FPS), Table 2 ablation(5 컴포넌트), Table 3 teacher-student 조합 전략, Table 4 합성 이상 비율 민감도, Table 5 효율성 비교

---

### Paper 4: NRdetector

| 항목 | 내용 |
|-----|------|
| 제목 | Noise-Resilient Point-wise Anomaly Detection in Time Series Using Weak Segment Labels |
| 저자 | Yaxuan Wang, Hao Cheng, Jing Xiong, Qingsong Wen, Han Jia, Ruixuan Song, Liyuan Zhang, Zhaowei Zhu, Yang Liu (9인) |
| Venue | **KDD 2025** (Proceedings of the 31st ACM SIGKDD, Toronto, August 3–7, 2025) |
| DOI | 10.1145/3690624.3709257 |
| arXiv | 2501.11959 |
| 선정 사유 | KDD 2025, PU learning + 시계열 이상탐지 + 동일 평가지표(VUS/PA%K/Affiliation), R16/R19/R20 직결 |
| 검증 상태 | arXiv HTML v1 전수 확인 완료 (상세: NRDETECTOR_DOSSIER.md) |

**섹션 구조**: Introduction → Related Work[TSAD / Learning with Noisy Labels / PU Learning] → Preliminaries → Methodology[§4.1 Temporal Embedding / Stage-1: §4.2 / Stage-2: §4.3] → Experiments[§5.1–5.5] → Conclusion + Appendix

**Figures**: Table 1 데이터셋 통계, Table 2 메인 성능(13 baselines), Table 3 정밀 비교(11 지표), Table 4 라벨 노이즈율 sweep, Table 5 Sample Selector ablation, Table 6 PU Criterion ablation, Fig.3 하이퍼파라미터 민감도

---

### Paper 5: CATCH

| 항목 | 내용 |
|-----|------|
| 제목 | CATCH: Channel-Aware multivariate Time Series Anomaly Detection via Frequency Patching |
| 저자 | Xingjian Wu, Xiangfei Qiu, Zhengyu Li, Yihang Wang, Jilin Hu, Chenjuan Guo, Hui Xiong, Bin Yang |
| Venue | **ICLR 2025** |
| arXiv | 2410.12261 |
| 선정 사유 | ICLR 2025 accepted, 주파수 도메인 패칭 + 채널 fusion, **22개 데이터셋(10 real-world + 12 synthetic)** SOTA (arXiv abstract 직접 재확인: "Extensive experiments on 10 real-world datasets and 12 synthetic datasets" — fixer r2, V-001 정정: 초판 "24개"는 근거 없음), 최신 방법론 비교 기준 |
| 검증 상태 | arXiv HTML 확인 완료 (2026-06-11) |

**섹션 구조**: Introduction → Related Work[§2.1 MTSAD / §2.2 Channel Strategies / §2.3 Frequency Domain] → CATCH[§3.1 Structure Overview / §3.2 Channel Fusion Module / §3.3 Time-Frequency Reconstruction / §3.4 Bi-level Optimization / §3.5 Anomaly Scoring] → Experiments[§4.1 Main Results / §4.2 Model Analysis] → Conclusion + Appendices A–D

**Figures**: Fig.1 주파수 밴드 이상 시각화, Fig.2 CATCH 아키텍처, Fig.3 이상 점수 계산, Fig.4 파라미터 민감도(4가지), Fig.5 이상 점수 시각화(5 anomaly type)

**Tables**: Table 1 기존 방법 속성 비교, Table 2 메인 성능(**10 real-world + 6 synthetic 이상 유형** — Table 2 캡션 직접 재확인: "Average A-R (AUC-ROC) and Aff-F (Affiliated-F1) accuracy measures for 10 real-world datasets and 6 synthetic datasets of different types of anomalies"; abstract의 총 커버리지 22개와 메인 테이블 항목 수는 다름 — fixer r2, V-001/S-001 연계 정정), Table 3 ablation(channel correlation / 최적화 / patching / scoring)

---

### Paper 6: Sub-Adjacent Transformer (IJCAI 2024)

| 항목 | 내용 |
|-----|------|
| 제목 | Sub-Adjacent Transformer: Improving Time Series Anomaly Detection with Reconstruction Error from Sub-Adjacent Neighborhoods |
| 저자 | Wenzhen Yue, Xianghua Ying, Ruohao Guo, DongDong Chen, Ji Shi, Bowei Xing, Yuqing Zhu, Taiyan Chen |
| Venue | **IJCAI 2024** |
| arXiv | 2404.18948 |
| 선정 사유 | IJCAI 2024, unsupervised TSAD, Anomaly Transformer와 직접 연관(adjacent-concentration 아이디어 계승·확장), 6 벤치마크 SOTA |
| 검증 상태 | arXiv HTML 확인 완료 (2026-06-11) |

**섹션 구조**: Introduction → Related Works[TSAD / Linear Attention] → Methods[§3.1 Problem Formulation / §3.2 Sub-Adjacent Neighborhoods / §3.3 Loss & Anomaly Score] → Experiments[Datasets / Implementation / Main Results / Ablation(5개)] → Conclusion + Appendix A–F

**Figures**: Fig.1 sub-adjacent neighborhood 개념도, Fig.2 attention contribution 개념, Fig.3 attention matrix 시각화, Fig.4 linear vs vanilla attention, Fig.5 NeurIPS-TS 시각화

**Tables**: Table 1 데이터셋 통계, Table 2 실세계 6 데이터셋 성능(15 baselines), Table 3 NeurIPS-TS 합성 성능, Table 4–8 ablation 및 민감도

---

### Paper 7: TSINR (KDD 2025)

| 항목 | 내용 |
|-----|------|
| 제목 | TSINR: Capturing Temporal Continuity via Implicit Neural Representations for Time Series Anomaly Detection |
| 저자 | Mengxuan Li, Ke Liu, Hongyang Chen, Jiajun Bu, Hongwei Wang, Haishuai Wang |
| Venue | **KDD 2025 (SIGKDD)** |
| arXiv | 2411.11641 |
| 선정 사유 | KDD 2025, **INR(implicit neural representation) 기반 재구성이 주된 기여**(spectral bias로 저주파 우선 학습 → 고주파/불연속 이상에 민감) + LLM은 이상 변동 증폭의 **보조 컴포넌트**(abstract: "leverage a pre-trained large language model to amplify the intense fluctuations in anomalies" — fixer r2, V-003 정정: 초판은 보조 특징만 강조해 핵심 기여 누락), 최신 방법론 트렌드 파악 |
| 검증 상태 | arXiv 추상 확인 (2026-06-11); Phase 4 재검증 필요 |

---

### Paper 8: ModernTCN

| 항목 | 내용 |
|-----|------|
| 제목 | ModernTCN: A Modern Pure Convolution Structure for General Time Series Analysis |
| 저자 | Donghao Luo, Xue Wang |
| Venue | **ICLR 2024 (Spotlight)** |
| OpenReview | openreview.net/forum?id=vpJMJerXHU |
| 선정 사유 | ICLR 2024 spotlight, 시계열 분석 5개 태스크(이상탐지 포함) SOTA, CATCH의 2024 SOTA baseline으로 인용됨 |
| 검증 상태 | OpenReview 확인 완료 (2026-06-11); 상세 구조는 Phase 4 재검증 필요 |

---

### Paper 9: PatchAD

| 항목 | 내용 |
|-----|------|
| 제목 | PatchAD: A Lightweight Patch-based MLP-Mixer for Time Series Anomaly Detection |
| 저자 | Zhijie Zhong, Zhiwen Yu, Yiyuan Yang, Weizheng Wang, Kaixiang Yang |
| Venue | arXiv 2401.09793 (under review 2025 기준); 최종 venue 미확정 |
| arXiv | 2401.09793 |
| 선정 사유 | 30개 이상 baseline 비교, patch-based MLP-Mixer 아키텍처, 대규모 구조 분석 유용. contrastive learning 기반. **Phase 4에서 venue 재확인 필수** |
| 검증 상태 | arXiv HTML 확인 완료; venue 미확정 주의 |

**섹션 구조**: Introduction → Preliminaries → Proposed Method[5개 소절] → Experiments[Setups / Comparison / Model Analysis] → Mechanism & Theory → Conclusion + 광범위 Appendix

**총 페이지**: 24p, 16 figures, 13 tables

---

### Paper 10: DACR

| 항목 | 내용 |
|-----|------|
| 제목 | DACR: Distribution-Augmented Contrastive Reconstruction for Time-Series Anomaly Detection |
| 저자 | Lixu Wang, Shichao Xu, Xinyu Du, Qi Zhu |
| Venue | **ICASSP 2024** (IEEE International Conference on Acoustics, Speech and Signal Processing) |
| arXiv | 2401.11271 |
| 선정 사유 | 2024 단편 학회, contrastive + reconstruction 결합, 9 벤치마크, 최신 트렌드 |
| 검증 상태 | arXiv 추상 확인; Phase 4 재검증 필요 |

---

### Paper 11: MAE (Masked Autoencoder) — 기반 방법론 참조

| 항목 | 내용 |
|-----|------|
| 제목 | Masked Autoencoders Are Scalable Vision Learners |
| 저자 | Kaiming He, Xinlei Chen, Saining Xie, Yanghao Li, Piotr Dollár, Ross Girshick |
| Venue | **CVPR 2022** |
| arXiv | 2111.06377 |
| 선정 사유 | MAE 원류 논문. TSMAE의 backbone 기반. 아키텍처 asymmetric encoder-decoder, 75% masking. 가장 많이 인용되는 기반 논문. |
| 검증 상태 | arXiv 확인 완료 (2026-06-11) |

---

### Paper 12: DTAAD

| 항목 | 내용 |
|-----|------|
| 제목 | DTAAD: Dual Tcn-Attention Networks for Anomaly Detection in Multivariate Time Series Data |
| 저자 | Lingrui Yu |
| Venue | **Knowledge-Based Systems**, Vol.295, 2024, pp.111849 (Elsevier) |
| arXiv | 2302.10753 |
| 선정 사유 | Elsevier 저널 구조 관례 파악 목적. TCN+Attention 결합, 7 데이터셋, Elsevier 전형적 레이아웃 |
| 검증 상태 | arXiv HTML 확인; Elsevier 최종본과 차이 있을 수 있음 — Phase 4 재검증 필요 |

---

### Paper 13: MEMTO

| 항목 | 내용 |
|-----|------|
| 제목 | MEMTO: Memory-guided Transformer for Multivariate Time Series Anomaly Detection |
| 저자 | Junho Song, Keonwoo Kim, Jeonglyul Oh, Sungzoon Cho |
| Venue | **NeurIPS 2023** (Advances in Neural Information Processing Systems 36, Main Conference Track) |
| arXiv | 2312.02530 |
| 선정 사유 | Memory module + K-means 초기화 + bi-dimensional scoring 구조 분석 |
| 검증 상태 | **NeurIPS 2023 proceedings 직접 확인** (papers.nips.cc/paper_files/paper/2023/hash/b4c898eb1fb556b8d871fbe9ead92256 — 제목·저자 4인 일치, 2026-06-11; fixer r2, V-002 정정: 초판 "venue 미확인, 2024 추정"은 오류였고 SENTENCE_CORPUS 로스터의 "NeurIPS 2023"과 모순이었음 — 본 정정으로 문서 간 정합) |

---

### Paper 14: DDMT

| 항목 | 내용 |
|-----|------|
| 제목 | DDMT: Denoising Diffusion Mask Transformer Models for Multivariate Time Series Anomaly Detection |
| 저자 | Chaocheng Yang, Tingyin Wang, Xuanhui Yan |
| Venue | arXiv 2310.08800 (venue 미확인) |
| arXiv | 2310.08800 |
| 선정 사유 | 확산 모델 + Transformer 결합 최신 트렌드. 마스킹 + 확산 접근법 구조 비교. **Phase 4 venue 확인 필수** |
| 검증 상태 | arXiv 추상 확인; venue 미확정 주의 |

---

## 4. 미수록 후보 (Phase 4에서 확인 후 추가 가능)

다음 논문들은 관련성이 높으나 현 단계에서 venue·구조 정보가 부족하여 미수록:

| 논문 | 이유 | 후속 조치 |
|-----|------|---------|
| OmniAnomaly (KDD 2019) | venue 확인됨, 구조 미파악 | Phase 4 확인 |
| TranAD (VLDB 2022) | venue 확인, 구조 미파악 | Phase 4 확인 |
| TimesNet (ICLR 2023) | venue 확인, TSAD 평가 포함 | Phase 4 확인 |
| CrossAD / OracleAD (NeurIPS 2025 계열) | venue 미확인 | Phase 4 확인 |
| AnomalyBERT (ICLR 2023 workshop) | workshop 논문 | Phase 4 확인 |
| **SARAD** (Dai, He, Yang, Leeke) | **NeurIPS 2024 Main Track 확인** (proceedings.neurips.cc/paper_files/paper/2024/hash/56ad264a…, 2026-06-11 — fixer r2, V-004) — Spatial Association-Aware AD for MTS; 구조 미파악 | Phase 4 확인 |
| **"The Elephant in the Room: Towards A Reliable Time-Series Anomaly Detection Benchmark"** (Liu & Paparrizos) | **NeurIPS 2024 Datasets & Benchmarks Track 확인** (nips.cc/virtual/2024/poster/97690, 2026-06-11 — fixer r2, V-004) — 40 데이터셋·40 알고리즘 TSAD 벤치마크, **VUS-PR을 최신뢰 지표로 권고** → 우리 평가지표 정당화 인용 후보 | Phase 4 확인 (인용 우선순위 높음) |

---

## 5. 선정 논문 venue 분포 요약

| venue | 논문 수 | 논문 |
|-------|--------|-----|
| ICLR (2022–2025) | 3 | Anomaly Transformer, ModernTCN, CATCH |
| KDD (2023–2025) | 3 | DCdetector, NRdetector, TSINR |
| CVPR (2022, 2024) | 2 | MAE, SDMAE |
| NeurIPS 2023 | 1 | MEMTO (fixer r2, V-002 — proceedings 확인으로 "venue 미확정"에서 이동) |
| IJCAI 2024 | 1 | Sub-Adjacent Transformer |
| ICASSP 2024 | 1 | DACR |
| Elsevier KBS 2024 | 1 | DTAAD |
| arXiv (venue 미확정) | 2 | PatchAD, DDMT |

**시계열 이상탐지 직접 대상 논문**: 11편 (Paper 1–2, 4–9, 12–14)
**관련 기반/방법론 논문**: 3편 (Paper 3, 11 + NRdetector)

---

## 부록: 정정 이력

### 2026-06-11 fixer r2 (adversarial review `paper/99_reviews/p2_venue_corpus_r1.md` 반영; fixlog: `p2_fixlog_r2.md`)

1. **[V-001, MAJOR]** Paper 5 CATCH 선정 사유 "24개 데이터셋 SOTA" → "22개(10 real-world + 12 synthetic)" — arXiv abstract(2410.12261) 직접 재확인. Tables 행도 메인 테이블 실측(10 real + 6 synthetic 이상 유형 × 2지표)으로 정정.
2. **[V-002, MAJOR / C-001 연계]** Paper 13 MEMTO venue "미확인, 2024 추정" → **NeurIPS 2023** — papers.nips.cc proceedings 직접 확인(제목·저자 일치). SENTENCE_CORPUS 로스터와의 모순 해소. §5 venue 분포 표 갱신(arXiv 미확정 3→2).
3. **[V-003, MINOR]** Paper 7 TSINR 선정 사유 — INR 기반 재구성을 주된 기여로 명시, LLM은 보조 컴포넌트로 강등 (arXiv abstract 재확인).
4. **[V-004, MINOR]** NeurIPS 2024 TSAD 논문 직접 조회 — SARAD(Main Track)·TSB-AD Elephant in the Room(D&B Track, VUS-PR 권고) 확인, §4 미수록 후보에 등재. STRUCTURE §I.4 주의사항도 동기 갱신.
5. **[V-005, NOTE]** Paper 1 Anomaly Transformer 선정 사유 — abstract "six benchmarks" 재확인, 벤치마크 총 6개(+NeurIPS-TS) 명시.
