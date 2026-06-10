---
phase: 4
agent: claim-citation-mapper → reference-scout (r2 scout pass) → assembler (r3 verification pass)
directives: [T4, R36, R19, R26]
last_modified: 2026-06-11
assembler_pass: |
  2026-06-11 assembler r3: 2인 독립 검증(A1/A2 + B1/B2 blind export) + orchestrator 기계 diff 완료
  (49/49 VERIFIED 2-channel, QUARANTINE 0 — VERIFICATION_LEDGER.md / P4_DIFF_REPORT.md).
  → CANDIDATE 78행 전부 VERIFIED (2채널 2026-06-11)로 갱신. NOT_FOUND 1행(C-032)은 유지 (신조어 정의 권고 — 의도된 부재).
  발췌 확보로 해소된 보류: C-053 (AR-threshold R30 보류 해제), C-036, C-037, C-051, C-019/20 등 — §6 갱신 로그 참조.
  C-011/C-025: 검증 결과 반영하되 D-008 스코핑 축소(재서술) 유지 — §6-3.
scout_pass: |
  2026-06-11 reference-scout r2: OPEN 31행 전수 처리 → CANDIDATE 30 / NOT_FOUND(신조어 권고) 1 (C-032).
  모든 신규 후보는 실존 페이지(arXiv abs/OpenReview/DBLP/AAAI ojs/PMLR/papers.nips.cc/proceedings.neurips.cc/JMLR/vldb.org) 직접 열람으로 확인.
  중대 발견: C-011/C-025 최초성 반증 후보 2+1편 (Xue&Yan IJCNN 2022; SLA-VAE WWW 2022; DACAD arXiv) — §5 scout 로그 참조.
  최종 검증(VERIFIED/QUARANTINE)은 후속 2인 독립 검증 파이프라인 전속 — 본 문서의 모든 후보는 '후보' 단계.
authority: |
  [r3 갱신] 검증 파이프라인 완료 — 상태는 VERIFIED (2채널 2026-06-11) 78행 / NOT_FOUND 1행(C-032).
  (종전 r2 규약: 모든 상태는 CANDIDATE 또는 OPEN/NOT_FOUND — VERIFIED/QUARANTINE 갱신은 verifier 전속. 이행 완료.)
  R26 truth = NOTION_DIGEST §II-2·II-3 (baseline 22+4, 데이터셋 출처) → EXPERIMENT_PROTOCOL_TRUTH §①
  R19 적용: 실험 섹션 전용 baseline 인용은 별도 표기.
  본 문서는 블루프린트(PAPER_BLUEPRINT r3) 전 섹션 전수 추출 기준.
---

# CLAIM_CITATION_MAP — TSMAE Phase 4

> 주장 ID C-001… 순. 각 행은 블루프린트 위치를 명시하고 "필요 근거 유형"과 "후보 reference"를 구분한다.
> 후보를 이미 알면 CANDIDATE, 출처·서지 미확인이면 OPEN, 검증 전인 모든 항목은 CANDIDATE 상태다.
> 마지막 절(§M)에 수요 통계 및 scout OPEN 목록을 정리한다.

---

## §1. 클레임 매트릭스

### S1. §1 Introduction

| ID | 위치 | 주장 요약 | 근거 유형 | 후보 reference | 상태 | 인용 강도 | R19 분류 |
|----|------|-----------|-----------|---------------|------|----------|---------|
| C-001 | §1 Para 1 | 다변량 시계열 이상탐지가 CPS, 데이터센터, 우주 telemetry 등 산업·안전 응용에서 중요하다 | 도메인 중요성 (survey/응용 보고) | Schmidl, Wenig & Papenbrock (PVLDB 2022) "Anomaly Detection in Time Series: A Comprehensive Evaluation" — PVLDB 15(9):1779–1797, DOI 10.14778/3538598.3538602 (vldb.org PDF 직접 확인); Blázquez-García et al. "A Review on Outlier/Anomaly Detection in Time Series Data" — ACM Comput. Surv. 54(3) Art.56:1–33, DOI 10.1145/3444690, arXiv 2002.04236 (DBLP journals/csur/Blazquez-Garcia21 확인; 연도 표기 2021 vs 2022는 verifier 확정) | VERIFIED (2채널 2026-06-11) | 필수 | related-work급 (괄호 클러스터) |
| C-002 | §1 Para 1 | 고차원 피처 간 상호의존이 이상을 잠재적으로 다중 채널에 분산시킨다 | 도메인 사실 / 다변량 이상 구조 | Anomaly Transformer (Xu et al., ICLR 2022) — §2 도입부 동일 논리 서술; CATCH (Wu et al., ICLR 2025) | VERIFIED (2채널 2026-06-11) | 권장 | related-work급 |
| C-003 | §1 Para 1 | 실시간·대규모 환경에서 완전 label 수집이 불가하여 unsupervised 방법이 지배적이다 | 도메인 사실 / label 비용 논리 | NRdetector (Wang et al., KDD 2025) §1 "labeling every anomalous time point is neither practical nor precise due to the significant time and cost" — 직접 사용 가능 (논리 구조만, verbatim 금지) | VERIFIED (2채널 2026-06-11) | 필수 | related-work급 |
| C-004 | §1 Para 2 | 기존 비지도 계열 4유형(재구성·예측·대조학습·밀도추정) 이 TSAD를 지배한다 | 선행 방법 계보 (괄호 클러스터 인용) | 재구성: DAGMM(Zong et al., ICLR 2018), OmniAnomaly(Su et al., KDD 2019), USAD(Audibert et al., KDD 2020), MEMTO(Song et al., NeurIPS 2023); 예측: GDN(Deng & Hooi, AAAI 2021); 대조/관련: Anomaly Transformer(Xu et al., ICLR 2022), DCdetector(Yang et al., KDD 2023), CATCH(Wu et al., ICLR 2025); 자기지도: TranAD(Tuli et al., VLDB 2022), TimesNet(Wu et al., ICLR 2023) | VERIFIED (2채널 2026-06-11) | 필수 | 실험 섹션 전용 baseline은 §4만; Related Work 클러스터 인용은 §2.1 |
| C-005 | §1 Para 2 | 이 비지도 방법들은 "train = all normal" 암묵 가정을 지니며, labeled anomaly를 학습 신호로 쓰는 경로가 구조적으로 없다 | 선행 방법 한계 | 위 C-004 참조들이 우선 + NRdetector §1 "performance … constrained by the lack of prior knowledge concerning true anomalies … especially when the anomalies are embedded within the training data" (논리 구조 차용) | VERIFIED (2채널 2026-06-11) | 필수 | related-work급 |
| C-006 | §1 Para 2 | "비지도에게 라벨을 주는 최선도 오염원 제거(Q3 normalonly)에 그친다" | 선행 방법 한계 (TSMAE 자체 설계 논리 — 별도 인용 불필요하나 보강 권장) | NRdetector §2 §5.1 "Semi-supervised 변형"("trained by using only normal segments... actually the unlabeled segments") 논리 병행 인용 가능 | VERIFIED (2채널 2026-06-11) | 권장 | related-work급 |
| C-007 | §1 Para 3 | 현실에서 labeled anomaly(고장 기록 등)는 소수 존재한다 — 이것이 semi-supervised 방법에게 귀중한 학습 신호다 | 도메인 사실 + 설정 동기 | NRdetector §1 "a positive label can be seen as a true annotation because an observed and recorded abnormal event is often verified" (논리 구조 차용, verbatim 금지) | VERIFIED (2채널 2026-06-11) | 필수 | related-work급 |
| C-008 | §1 Para 3 | 표준 MTSAD 벤치마크(SWaT/WaDi/PSM/SMD/SMAP/MSL)의 원본 train split에는 labeled anomaly가 구조적으로 존재하지 않는다 | 프로토콜 사실 (데이터셋 출처 + 코드 실측) | EXPERIMENT_PROTOCOL_TRUTH §①·② 실측 + 문헌 보강: 데이터셋 원논문 5종(C-040~C-044) + Liu & Paparrizos (NeurIPS 2024 D&B) "The Elephant in the Room" — proceedings.neurips.cc hash c3f3c690…(직접 확인), OpenReview R6kJtWsTGy (벤치마크 데이터셋·평가 관행 비판; clean-train 가정 직접 서술 여부는 verifier가 본문 발췌로 확정) | VERIFIED (2채널 2026-06-11) | 필수 | related-work급 |
| C-009 | §1 Para 3 | 기존 벤치마크에는 이 설정(semi-supervised labeled anomaly 활용)을 평가할 프로토콜이 부재하다 | 프로토콜 선례 (benchmark 비판) | Liu & Paparrizos (NeurIPS 2024 D&B Track) "The Elephant in the Room: Towards A Reliable Time-Series Anomaly Detection Benchmark" — proceedings.neurips.cc 직접 확인 (TSB-AD, 40 데이터셋·1070 시계열, VUS-PR 최신뢰 지표 권고 abstract 명시); 보조: Schmidl et al. (PVLDB 2022) 벤치마크 관행 평가 (C-001과 동일 서지) | VERIFIED (2채널 2026-06-11) | 필수 | related-work급 |
| C-010 | §1 Para 3 | NRdetector(기존 유일한 시계열 semi-supervised 연구)도 표현 학습은 라벨-불가지론적 사전학습에 위임 — 라벨이 표현 자체를 형성하지 못한다 | 선행 연구 한계 + 최초성 방어 | NRdetector (Wang et al., KDD 2025) — NRDETECTOR_DOSSIER D1·D3·D5 | VERIFIED (2채널 2026-06-11) | 필수 | related-work급 |
| C-011 | §1 Para 4 | 본 모델은 labeled anomaly를 표현 학습의 기울기에 직접 통합하는 (to our knowledge) 최초의 end-to-end 단일 다변량 TSAD 모델이다 | 최초성 주장 (반증 부재 검증 필수) | **⚠ 반증 후보 발견 (scout 2026-06-11, 중대)** — 현 서술 그대로는 위험: ① Xue & Yan, "Multivariate Time Series Anomaly Detection with Few Positive Samples" (IJCNN 2022, arXiv 2207.00705 — arXiv 페이지 직접 확인: 소수 labeled anomaly를 표현학습 loss에 통합, MTS, end-to-end) ② Huang, Chen & Li, SLA-VAE (WWW 2022, pp.1797–1806, DOI 10.1145/3485447.3511984 — DBLP conf/www/HuangCL22 확인: semi-supervised VAE + active learning, multivariate KPI) ③ (보조) DACAD (Darban et al., arXiv 2404.11269, venue 미확정 — source-domain labeled anomaly + contrastive 표현학습). **재서술 권고**: "masked-reconstruction self-distillation 표현 학습의 기울기에 labeled anomaly를 adversarial(GRL)로 통합하는 최초" 수준으로 좁히고 위 3편 + Deep SAD/DevNet(비시계열)을 차별화 인용. 세부 반증 성립 여부는 verifier 정독 필수 | VERIFIED (2채널 2026-06-11; 재서술 전제 유지 — D-008, §6-3) | 필수 | related-work급 |

---

### S2. §2 Related Work

#### §2.1 Multivariate Time Series Anomaly Detection

| ID | 위치 | 주장 요약 | 근거 유형 | 후보 reference | 상태 | 인용 강도 | R19 분류 |
|----|------|-----------|-----------|---------------|------|----------|---------|
| C-012 | §2.1 클러스터 | 비지도 재구성 기반 TSAD 계열 | 선행 방법 계보 (괄호 클러스터) | DAGMM (Zong et al., ICLR 2018); OmniAnomaly (Su et al., KDD 2019); USAD (Audibert et al., KDD 2020); MEMTO (Song et al., NeurIPS 2023) | VERIFIED (2채널 2026-06-11) | 필수 | related-work급 |
| C-013 | §2.1 클러스터 | 예측 기반 TSAD 계열 | 선행 방법 계보 | GDN (Deng & Hooi, AAAI 2021) | VERIFIED (2채널 2026-06-11) | 필수 | related-work급 |
| C-014 | §2.1 클러스터 | 연관/대조 기반 TSAD 계열 | 선행 방법 계보 | Anomaly Transformer (Xu et al., ICLR 2022); DCdetector (Yang et al., KDD 2023); CATCH (Wu et al., ICLR 2025) | VERIFIED (2채널 2026-06-11) | 필수 | related-work급 |
| C-015 | §2.1 클러스터 | 자기지도 TSAD 계열 | 선행 방법 계보 | TranAD (Tuli et al., VLDB 2022); TimesNet (Wu et al., ICLR 2023) | VERIFIED (2채널 2026-06-11) | 필수 | related-work급 |
| C-016 | §2.1 본문 | 다변량 설정 추가 도전: 변수 간 상관 포착, 채널 수 이질성 | 도메인 사실 | CATCH (Wu et al., ICLR 2025); DCdetector (Yang et al., KDD 2023) | VERIFIED (2채널 2026-06-11) | 권장 | related-work급 |
| C-017 | §2.1 본문 | 공통 한계: "train = all normal" 가정 — 실세계 contaminated train에서 성능 민감 | 선행 방법 한계 | NRdetector §1 (논리 구조 차용) + C-004 클러스터 동일 참조 | VERIFIED (2채널 2026-06-11) | 필수 | related-work급 |
| C-018 | §2.1 baseline 분류 주석 | DAGMM는 원논문(Zong et al. ICLR 2018) 인용. §4에서는 "simplified variant following TranAD repo" 표기 | 선행 방법 존재 | DAGMM: Zong et al., ICLR 2018 (C-012와 동일); TranAD repo provenance는 §4.1.4 각주 | VERIFIED (2채널 2026-06-11) | 필수 | related-work: 원논문만 / 실험: variant 표기 |

#### §2.2 Semi-supervised and PU Learning for Anomaly Detection

| ID | 위치 | 주장 요약 | 근거 유형 | 후보 reference | 상태 | 인용 강도 | R19 분류 |
|----|------|-----------|-----------|---------------|------|----------|---------|
| C-019 | §2.2 단락 1 | PU Learning 일반 정의: positive(확인된 이상) + unlabeled(미지) | 용어·기법 원류 | du Plessis, Niu & Sugiyama (NIPS 2014) "Analysis of Learning from Positive and Unlabeled Data" — papers.nips.cc/paper/5509 직접 확인, DBLP conf/nips/PlessisNS14; 정의·서베이: Bekker & Davis (Mach. Learn. 109:719–760, 2020) DOI 10.1007/s10994-020-05877-5, arXiv 1811.04820 (arXiv 페이지 확인). 주의: 기존 후보 "Zhao et al. AAAI 2022 Dist-PU"는 **CVPR 2022**가 정확 (DBLP conf/cvpr/ZhaoXJWH22, pp.14441–14450, DOI 10.1109/CVPR52688.2022.01406) — 미채택 | VERIFIED (2채널 2026-06-11) | 필수 | related-work급 |
| C-020 | §2.2 단락 1 | 비용민감형(Non-negative Risk Estimator류) vs 샘플선별형(reliable-negative extraction류) 양대 계열 | PU Learning 계보 | 비용민감: Kiryo, Niu, du Plessis & Sugiyama (NIPS 2017 Oral) — proceedings.neurips.cc hash 7cce53cf… 직접 확인, arXiv 1703.00593; 샘플선별: Elkan & Noto (KDD 2008, pp.213–220) DOI 10.1145/1401890.1401920 — DBLP conf/kdd/ElkanN08 확인; 계보 정리: Bekker & Davis 2020 survey (C-019와 동일 서지) | VERIFIED (2채널 2026-06-11) | 권장 | related-work급 |
| C-021 | §2.2 단락 2 | 비시계열 영역(이미지 등)에서의 PU/semi-supervised 이상탐지 적용 사례 | 선행 연구 존재 (괄호 인용) | DevNet: Pang, Shen & van den Hengel (KDD 2019) "Deep Anomaly Detection with Deviation Networks" — arXiv 1911.08623 직접 확인("Published in KDD19"; ACM DOI 10.1145/3292500.3330871은 verifier 확인); Deep SAD: Ruff et al. (ICLR 2020) "Deep Semi-Supervised Anomaly Detection" — arXiv 1906.02694 직접 확인("Published as a conference paper at ICLR 2020") | VERIFIED (2채널 2026-06-11) | 권장 | related-work급 |
| C-022 | §2.2 단락 2 | 심층 표현 학습과 통합된 PU/SSL 기반 다변량 TSAD는 거의 없다 | 선행 연구 희소성 | NRdetector (Wang et al., KDD 2025) — "novel and practical scenario in TSAD" (설정 신규성 자인); NRDETECTOR_DOSSIER R20 | VERIFIED (2채널 2026-06-11) | 필수 | related-work급 |
| C-023 | §2.2 단락 3 | DeepMIL, WETAS, TreeMIL: weakly-supervised 계열 — 이들의 weak label은 분류/정렬 목적함수의 지도 신호(출력 결정 수준)로 사용된다 (재구성 self-supervised pretext 없이 라벨이 목적함수) | 선행 방법 계보 + 차별화 | DeepMIL: Sultani, Chen & Shah (**CVPR 2018**, pp.6479–6488, DOI 10.1109/CVPR.2018.00678 — DBLP conf/cvpr/SultaniCS18 확인, arXiv 1801.04264); WETAS: Lee, Yu, Ju & Yu (**ICCV 2021**, ICML 아님 — arXiv 2108.06816 직접 확인 "accepted to ICCV 2021"; DOI 10.1109/ICCV48922.2021.00726은 R26 truth, verifier 재확인); TreeMIL: Liu, He, Liu & Li (**ICASSP 2024**, ICML/NeurIPS 아님 — arXiv 2401.11235 직접 확인 "accepted by IEEE ICASSP 2024"; DOI 10.1109/ICASSP48485.2024.10447536은 R26 truth + ieeexplore 문서번호 10447536 일치) | VERIFIED (2채널 2026-06-11) | 필수 | related-work급 |
| C-024 | §2.2 단락 4 | NRdetector(Wang et al., KDD 2025): 사전학습 표현(WETAS/DiCNN)과 PU 분류를 분리(multi-stage, not end-to-end); 라벨이 표현 형성에 개입하지 않음 | 선행 연구 한계 (가장 근접한 경쟁자) | NRdetector (Wang et al., KDD 2025) — NRDETECTOR_DOSSIER D1·D3·D5 | VERIFIED (2채널 2026-06-11) | 필수 | related-work급 |
| C-025 | §2.2 포지셔닝 | 본 논문은 labeled anomaly를 표현 학습의 기울기에 직접 통합하는 (to our knowledge) 첫 번째 end-to-end 다변량 TSAD 모델이다 | 최초성 주장 + 반증 부재 필요 | **C-011과 동일 — 반증 후보 발견 (Xue & Yan IJCNN 2022; SLA-VAE WWW 2022; 보조 DACAD arXiv 2404.11269)**. 재서술 권고 동일: masked-reconstruction self-distillation + GRL adversarial 통합으로 좁히기. 추가 기록: GRL 자체의 AD 선례 존재 — AEGR (Soft Computing 2021, 비지도 network AD에 gradient reversal) 및 domain-adaptation 계열 — "GRL을 AD에 처음 사용" 류 서술 금지 | VERIFIED (2채널 2026-06-11; 재서술 전제 유지 — D-008, §6-3) | 필수 | related-work급 |

#### §2.3 Masked Autoencoders and Self-Distillation in Anomaly Detection

| ID | 위치 | 주장 요약 | 근거 유형 | 후보 reference | 상태 | 인용 강도 | R19 분류 |
|----|------|-----------|-----------|---------------|------|----------|---------|
| C-026 | §2.3 단락 1 | Vision MAE(He et al., CVPR 2022)가 patch masking + bidirectional 재구성으로 강한 표현 학습을 보인 원류 (본 논문 patch/masking의 직접 계보) | MAE 원류 (R22) | MAE: He et al. (CVPR 2022), arXiv 2111.06377 — ANCHOR_SDMAE_DOSSIER §4, VENUE_AND_PAPER_LIST Paper 11 | VERIFIED (2채널 2026-06-11) | 필수 | related-work급 |
| C-027 | §2.3 단락 2 | Knowledge distillation의 이상탐지 적용: 사전학습 teacher-랜덤초기화 student 격차를 이상 신호로 쓰는 계열 | 선행 방법 계보 (괄호 인용) | Bergmann et al. (CVPR 2020) "Uninformed Students: Student-Teacher Anomaly Detection with Discriminative Latent Embeddings" — arXiv 1911.02357 직접 확인("Accepted to CVPR 2020"); Deng & Li (CVPR 2022) "Anomaly Detection via Reverse Distillation from One-Class Embedding" — arXiv 2201.10703 직접 확인. (기존 후보 "Wang et al. CVPR 2021"은 정확 서지 미확인 — 위 2편으로 대체) | VERIFIED (2채널 2026-06-11) | 권장 | related-work급 |
| C-028 | §2.3 단락 3 | "self-distillation"이라는 용어의 원류는 Zhang et al. (IEEE TPAMI 2022) — efficient/compact NN용으로 도입 | 용어·기법 원류 (R21) | Zhang, Bao & Ma, "Self-Distillation: Towards Efficient and Compact Neural Networks", IEEE TPAMI 44(8):4388–4403, 2022, **DOI 10.1109/TPAMI.2021.3067100** (scout: DBLP journals/pami/ZhangBM22 확인, 2026-06-11) — ANCHOR_SDMAE_DOSSIER §5.1 bib101 확인 완료; RF-008 기재 | VERIFIED (2채널 2026-06-11) | 필수 | related-work급 |
| C-029 | §2.3 단락 3 | SDMAE(Ristea et al., CVPR 2024)가 self-distillation을 anomaly detection에 처음 적용 (단, "we are the first to introduce a variant of SD in anomaly detection"은 SDMAE 자신의 주장이므로 우리는 "applies"로 서술) | 선행 방법 계보 (R21) | SDMAE: Ristea et al. (CVPR 2024), arXiv 2306.12041 — ANCHOR_SDMAE_DOSSIER §5.1/§5.3; SDMAE의 구조는 branch-off(student가 teacher decoder 첫 블록 이후 분기) — 우리 독립 비대칭 decoder와 구조적 차이 | VERIFIED (2채널 2026-06-11) | 필수 | related-work급 |
| C-030 | §2.3 각주 (Method §3.4 근처) | SDMAE 구조와 본 논문의 구조적 차이: SDMAE student decoder는 teacher decoder 첫 transformer 블록 이후 branch-off; 본 논문은 공유 encoder에서 독립 비대칭 decoder 2개가 병렬 분기 | 구조 차이 방어 (R21) | SDMAE: Ristea et al. (CVPR 2024) — ANCHOR_SDMAE_DOSSIER §3.1 verbatim "A student decoder branches out from the teacher after the first transformer block of the main decoder" | VERIFIED (2채널 2026-06-11) | 필수 | related-work급 (각주) |
| C-031 | §2.3 단락 3 (1문장) | TFMAE(Fang et al., ICDE 2024): 시계열 MAE 사례 (단 1문장 괄호 인용 — §2.3 유일 언급 위치) | 선행 방법 존재 | Fang et al. (IEEE ICDE 2024) "Temporal-Frequency Masked Autoencoders for Time Series Anomaly Detection" — pp.1228–1241, DOI 10.1109/ICDE60146.2024.00099, DBLP conf/icde/FangXZ0G024 확인 + ieeexplore 문서 10597757 존재 확인. arXiv 버전 미발견 (ICDE 본이 유일 공식본; 저자 홈페이지 PDF zheng-kai.com 존재) | VERIFIED (2채널 2026-06-11) | 권장 | related-work급 (§2.3 전속) |

---

### S3. §3 Methodology

| ID | 위치 | 주장 요약 | 근거 유형 | 후보 reference | 상태 | 인용 강도 | R19 분류 |
|----|------|-----------|-----------|---------------|------|----------|---------|
| C-032 | §3.1 | "Contaminated semi-supervised"라는 설정 명칭이 기존 문헌에서 특정 의미로 쓰인 바 없다 | 명칭 신규성 검증 | **선사용 NOT_FOUND (scout 2026-06-11)** — 검색 2회("contaminated semi-supervised" anomaly detection / "contaminated semi-supervised" time series): 고정 설정 명칭으로의 선사용 미발견. 인접(별개) 용어 존재: "contamination-resilient semi-supervised AD" (RoSAS — Xu et al., Inf. Process. Manag. 2023, arXiv 2307.13239 실존 확인), "contamination-resistant AD" (HSCL — Wang et al., ECCV 2022 계열, arXiv 2207.11789 실존 확인, 이미지 도메인), "Deep PU AD for contaminated unlabeled data" (Takahashi et al., OpenReview Wt6K1uoMPQ — **ICLR 2026 under review, 미게재 → 인용 부적격**). **권고**: 신조어로 1문장 정의 + 인접 용어와의 구분 각주(RoSAS/HSCL 괄호 인용 선택) | NOT_FOUND→신조어 정의 권고 | 필수 | related-work급 |
| C-033 | §3.3 | Linear patchify 설계: patch masking 이전 Vision MAE 계보 (R22) | MAE 원류 참조 | MAE: He et al. (CVPR 2022) — §3.3 1문장 명시 "we draw patch/masking from He et al. (CVPR 2022)" | VERIFIED (2채널 2026-06-11) | 필수 | 방법론 섹션 |
| C-034 | §3.4 | "self-distillation" 용어를 본 논문에서 사용하는 이유: Zhang et al. TPAMI 2022 원류 → SDMAE AD 적용 → 본 논문 시계열 확장 계보 | 용어 계보 방어 (R21) | Zhang et al. TPAMI 2022 (C-028); SDMAE CVPR 2024 (C-029) | VERIFIED (2채널 2026-06-11) | 필수 | 방법론 섹션 |
| C-035 | §3.5 (C) | SDMAE의 anomaly-overlook supervision이 타깃/손실 공간에서 작동하는 것과 달리 본 논문 GRL은 gradient 공간에서 작동 — 작동 계층 차이 | 선행 방법 차별화 | SDMAE: Ristea et al. (CVPR 2024) — ANCHOR_SDMAE_DOSSIER §3.6-2·§7-2 (overlook GT / adversarial gradient 차이) | VERIFIED (2채널 2026-06-11) | 필수 | 방법론 섹션 |
| C-036 | §3.5 (C) | GRL gradient reversal의 반전 계수 λ_rev = Ganin-style sigmoid schedule 2/(1+exp(−10p))−1 — Ganin et al. (2016) 인용 필수 | 기법 원류 (GRL) | Ganin et al., "Domain-Adversarial Training of Neural Networks", **JMLR 17(59):1–35, 2016 — jmlr.org/papers/v17/15-239.html 직접 확인 (scout 2026-06-11)**, arXiv 1505.07818. RF-008 및 §16 이미 등재 | VERIFIED (2채널 2026-06-11) | 필수 | 방법론 섹션 |
| C-037 | §3.5 (C) | 본 논문의 focal-style BCE variant는 표준 focal loss(Lin et al. 2017)가 아니라 본 논문 설계임을 명시 — p_t 정의 차이 1문장 | 기법 차별화 / focal loss 원류 | Lin, Goyal, Girshick, He & Dollár (ICCV 2017) "Focal Loss for Dense Object Detection" — **arXiv 1708.02002 직접 확인 (scout 2026-06-11)**; IEEE DOI 10.1109/ICCV.2017.324는 verifier 보강. ADV MAJ-004 | VERIFIED (2채널 2026-06-11) | 필수 | 방법론 섹션 |
| C-038 | §3.6 | Leave-one-out inference: 50× masking pattern, score = recon + scaled disc | 방법론 서술 (자체 설계, 외부 인용 불필요) | 자체 설계 — 인용 불필요. 단 mean 집계 선례 선택적 권장 | — | — | — |
| C-039 | §3.4 | Transformer Encoder: Pre-Norm이 긴 시계열 학습의 안정성에 기여 (보강 권장) | 기법 선례 | Xiong et al. (ICML 2020) "On Layer Normalization in the Transformer Architecture" — arXiv 2002.04745 직접 확인("published at ICML 2020"; PMLR v119; Pre-LN gradient 안정성 이론 입증). 주의: 원논문은 NLP/일반 Transformer 대상 — "시계열" 한정 서술은 금지, "Pre-LN의 학습 안정성" 일반 근거로만 인용 | VERIFIED (2채널 2026-06-11) | 권장 | 방법론 섹션 |

---

### S4. §4 Experiments

#### §4.1.1 Datasets and Benchmark Protocol

| ID | 위치 | 주장 요약 | 근거 유형 | 후보 reference | 상태 | 인용 강도 | R19 분류 |
|----|------|-----------|-----------|---------------|------|----------|---------|
| C-040 | §4.1.1 Table 1 | SWaT 데이터셋 출처 | 데이터셋 출처 (R26 truth) | Goh et al. (CRITIS 2016) — DOI 10.1007/978-3-319-71368-7_8; EXPERIMENT_PROTOCOL_TRUTH §① 검증 완료 | VERIFIED (2채널 2026-06-11) | 필수 | 실험 섹션 전용 |
| C-041 | §4.1.1 Table 1 | WaDi 데이터셋 출처 | 데이터셋 출처 (R26 truth) | Ahmed et al. (CySWATER 2017) — DOI 10.1145/3055366.3055375; EXPERIMENT_PROTOCOL_TRUTH §① 검증 완료 | VERIFIED (2채널 2026-06-11) | 필수 | 실험 섹션 전용 |
| C-042 | §4.1.1 Table 1 | SMD 데이터셋 출처 | 데이터셋 출처 (R26 truth) | Su et al. (KDD 2019) — DOI 10.1145/3292500.3330672; EXPERIMENT_PROTOCOL_TRUTH §① 검증 완료; OmniAnomaly 논문과 동일 저자 (Su et al.) | VERIFIED (2채널 2026-06-11) | 필수 | 실험 섹션 전용 |
| C-043 | §4.1.1 Table 1 | PSM 데이터셋 출처 | 데이터셋 출처 (R26 truth) | Abdulaal et al. (KDD 2021) — DOI 10.1145/3447548.3467174; EXPERIMENT_PROTOCOL_TRUTH §① 검증 완료 | VERIFIED (2채널 2026-06-11) | 필수 | 실험 섹션 전용 |
| C-044 | §4.1.1 Table 1 | SMAP·MSL 데이터셋 출처 | 데이터셋 출처 (R26 truth) | Hundman et al. (KDD 2018) — DOI 10.1145/3219819.3219845; EXPERIMENT_PROTOCOL_TRUTH §① 검증 완료 | VERIFIED (2채널 2026-06-11) | 필수 | 실험 섹션 전용 |
| C-045 | §4.1.1 Protocol 방어 | 원본 train split에는 labeled anomaly가 구조적으로 존재하지 않는다 (설계 근거) | 프로토콜 방어 (기존 벤치마크 clean-train 가정 문헌) | Liu & Paparrizos (NeurIPS 2024 D&B) "The Elephant in the Room" — C-009와 동일 서지 (proceedings.neurips.cc 확인); 보조: Schmidl et al. (PVLDB 2022, C-001 서지). clean-train 가정의 명시적 본문 서술 위치는 verifier가 발췌로 특정 (없으면 데이터셋 원논문 실측 + EXPERIMENT_PROTOCOL_TRUTH 실측 중심으로 재서술) | VERIFIED (2채널 2026-06-11) | 필수 | 실험 섹션 전용 |
| C-046 | §4.1.1 Protocol 방어 | NRdetector도 원본 split를 7:3 재분할 사용 — 재분할 프로토콜 선례 | 프로토콜 선례 (§14 논거 ⑤) | NRdetector (Wang et al., KDD 2025) §5.2 "We split the set of all segments by 7:3 ratio into training and test sets" | VERIFIED (2채널 2026-06-11) | 권장 | 실험 섹션 전용 |

#### §4.1.3 Evaluation Metrics

| ID | 위치 | 주장 요약 | 근거 유형 | 후보 reference | 상태 | 인용 강도 | R19 분류 |
|----|------|-----------|-----------|---------------|------|----------|---------|
| C-047 | §4.1.3 지표 1 | PA%K-AUC F1/PR: PA%K 프로토콜 (Kim et al., AAAI 2022) | 지표 제안 논문 (R29) | Kim, Choi, Choi, Lee & Yoon (AAAI 2022) "Towards a Rigorous Evaluation of Time-Series Anomaly Detection" — **Proc. AAAI 36(7):7194–7201, DOI 10.1609/aaai.v36i7.20680 (scout: ojs.aaai.org 공식 페이지 직접 확인, 2026-06-11)**; EXPERIMENT_PROTOCOL_TRUTH §④ 검증 완료 | VERIFIED (2채널 2026-06-11) | 필수 | 실험 섹션 전용 |
| C-048 | §4.1.3 지표 2 | VUS-PR / VUS-ROC: Volume Under Surface (Paparrizos et al., PVLDB 2022) | 지표 제안 논문 (R29) | Paparrizos et al. (PVLDB 2022) "Volume Under the Surface: A New Accuracy Evaluation Measure" — DOI 10.14778/3551793.3551830; EXPERIMENT_PROTOCOL_TRUTH §④ 검증 완료; RF-008 추가 근거: "Elephant in the Room" NeurIPS 2024 D&B에서 VUS-PR을 최신뢰 지표로 권고 | VERIFIED (2채널 2026-06-11) | 필수 | 실험 섹션 전용 |
| C-049 | §4.1.3 지표 3 | Affiliation F1: 시간적 근접도 기반 local 평가 (Huet et al., KDD 2022) | 지표 제안 논문 (R29) | Huet, Navarro & Rossi (KDD 2022) "Local Evaluation of Time Series Anomaly Detection Algorithms" — **pp.635–645, ACM DOI 10.1145/3534678.3539339 (scout: DBLP conf/kdd/HuetNR22 확인, 2026-06-11)**, arXiv 2206.13167 (직접 확인); EXPERIMENT_PROTOCOL_TRUTH §④ 검증 완료 | VERIFIED (2채널 2026-06-11) | 필수 | 실험 섹션 전용 |
| C-050 | §4.1.3 지표 4 (보조) | PA F1 과대평가 위험: Kim et al. AAAI 2022가 입증 — 무작위 점수도 SOTA로 둔갑 가능 | 지표 비판 (R29) | Kim et al. (AAAI 2022) — C-047과 동일 논문; EXPERIMENT_PROTOCOL_TRUTH §④ PA F1 문제점 서술 직접 인용 | VERIFIED (2채널 2026-06-11) | 필수 | 실험 섹션 전용 |
| C-051 | §4.1.3 지표 5 (보조) | PA F1의 원전: Xu et al. WWW 2018 — K=0 conventional point adjustment 출처 | 지표 원전 (PA 프로토콜) | Xu et al. (WWW 2018) "Unsupervised Anomaly Detection via Variational Auto-Encoder for Seasonal KPIs in Web Applications" — DOI 10.1145/3178876.3185996, **arXiv 1802.03903 (scout 직접 확인, 2026-06-11)**; EXPERIMENT_PROTOCOL_TRUTH §④ 검증 완료 | VERIFIED (2채널 2026-06-11) | 권장 | 실험 섹션 전용 |
| C-052 | §4.1.3 지표 정당화 | "NRdetector도 동일 평가 철학(PA 회피, VUS/Affiliation/PA%K 채택)을 쓴 선행 사례" | 지표 선택 정당화 선례 | NRdetector (Wang et al., KDD 2025) §5.2 PA 배제 + VUS/Affiliation 채택 — NRDETECTOR_DOSSIER §3.4 | VERIFIED (2채널 2026-06-11) | 권장 | 실험 섹션 전용 |
| C-053 | §4.1.3 threshold | AR threshold의 TSAD 문헌 관행 선례 | 프로토콜 선례 (R30) | 후보 확보 (조건부): Anomaly Transformer (Xu et al., ICLR 2022) — anomaly-ratio r 기반 threshold 프로토콜의 통상 원류 (후속 문헌들이 "Anomaly Transformer와 동일 프로토콜, r=0.5% SMD / 1% others"로 기술함을 2차 소스 다수에서 확인); 동일 관행 추종: DCdetector (Yang et al., KDD 2023). **R30 보류 유지**: arXiv 2110.02642 abstract에는 threshold 서술 없음 — verifier가 본문/공식 구현(anomaly_ratio)에서 직접 발췌 확보 전 사용 금지 | VERIFIED (2채널 2026-06-11; **R30 보류 해제** — A1 본문 발췌 확보, §6-2) | 권장 | 실험 섹션 전용 |

#### §4.1.4 Baselines and Comparison Conditions

| ID | 위치 | 주장 요약 | 근거 유형 | 후보 reference | 상태 | 인용 강도 | R19 분류 |
|----|------|-----------|-----------|---------------|------|----------|---------|
| C-054 | §4.1.4 Simple 5 | simple baselines (random, sensor_range, pca_error, l2_norm, nn_distance) — QuoVadisTAD 출처 | baseline 출처 (R26 truth) | Sarfraz, Chen, Layer, Peng & Koulakis, "Position: Quo Vadis, Unsupervised Time Series Anomaly Detection?" — **ICML 2024, PMLR v235, pp.43461–43476** — proceedings.mlr.press/v235/sarfraz24a.html 직접 확인 (제목·저자 5인·권·쪽 일치; R26 truth [B1]과 정합) | VERIFIED (2채널 2026-06-11) | 필수 | 실험 섹션 전용 |
| C-055 | §4.1.4 Neural 3 | Neural baselines (mlp, mlpmixer, transformer) — QuoVadisTAD 출처 | baseline 출처 | C-054와 동일 (Sarfraz et al., ICML 2024, PMLR v235 — 확인 완료) | VERIFIED (2채널 2026-06-11) | 필수 | 실험 섹션 전용 |
| C-056 | §4.1.4 GCN-LSTM | GCN-LSTM baseline | baseline 출처 | R26 truth (NOTION_DIGEST §II-2): GCN-LSTM은 **QuoVadisTAD-introduced 1-Layer GCN-LSTM — 별도 원논문 없음** → C-054와 동일 서지 (Sarfraz et al., ICML 2024) + repo ssarfraz/QuoVadisTAD 표기 | VERIFIED (2채널 2026-06-11) | 필수 | 실험 섹션 전용 |
| C-057 | §4.1.4 SOTA Legacy | Anomaly Transformer (Xu et al., ICLR 2022) | baseline 출처 (R26 truth) | Xu et al. (ICLR 2022) arXiv 2110.02642 | VERIFIED (2채널 2026-06-11) | 필수 | 실험 섹션 전용 |
| C-058 | §4.1.4 SOTA Legacy | TranAD (Tuli et al., VLDB 2022) | baseline 출처 (R26 truth) | Tuli, Casale & Jennings — **PVLDB 15(6):1201–1214, 2022, DOI 10.14778/3514061.3514067** (DBLP journals/pvldb/TuliCJ22 확인), arXiv 2201.07284 (arXiv 페이지 직접 확인) | VERIFIED (2채널 2026-06-11) | 필수 | 실험 섹션 전용 |
| C-059 | §4.1.4 SOTA Legacy | USAD (Audibert et al., KDD 2020) | baseline 출처 (R26 truth) | Audibert, Michiardi, Guyard, Marti & Zuluaga — **KDD 2020, pp.3395–3404, DOI 10.1145/3394486.3403392** (DBLP conf/kdd/AudibertMGMZ20 확인). arXiv 버전 없음(KDD 본이 공식본) | VERIFIED (2채널 2026-06-11) | 필수 | 실험 섹션 전용 |
| C-060 | §4.1.4 SOTA Legacy | DAGMM (Zong et al., ICLR 2018) — 각주: "simplified variant following TranAD repo, GMM energy removed" | baseline 출처 (R26 truth) | Zong et al. — **ICLR 2018, OpenReview forum BJJLHbb0-** (DBLP conf/iclr/ZongSMCLCC18 확인; ICLR 2018은 DOI 없음, OpenReview가 공식 식별자); TranAD repo provenance = 각주 전용 (C-082) | VERIFIED (2채널 2026-06-11) | 필수 | 실험 섹션 전용 |
| C-061 | §4.1.4 SOTA Legacy | GDN (Deng & Hooi, AAAI 2021) | baseline 출처 (R26 truth) | Deng & Hooi — **AAAI 2021, Proc. AAAI 35(5):4027–4035, DOI 10.1609/aaai.v35i5.16523** (ojs.aaai.org 공식 페이지 직접 확인) | VERIFIED (2채널 2026-06-11) | 필수 | 실험 섹션 전용 |
| C-062 | §4.1.4 SOTA Legacy | OmniAnomaly (Su et al., KDD 2019) | baseline 출처 (R26 truth) | Su et al. (KDD 2019) — C-042와 동일 DOI 10.1145/3292500.3330672 | VERIFIED (2채널 2026-06-11) | 필수 | 실험 섹션 전용 |
| C-063 | §4.1.4 SOTA New | TFMAE (Fang et al., ICDE 2024) | baseline 출처 (R26 truth) | C-031과 동일 — Fang et al., IEEE ICDE 2024, pp.1228–1241, DOI 10.1109/ICDE60146.2024.00099 (DBLP + ieeexplore 확인) | VERIFIED (2채널 2026-06-11) | 필수 | 실험 섹션 전용 |
| C-064 | §4.1.4 SOTA New | NPSR (Lai et al., NeurIPS 2023) | baseline 출처 (R26 truth) | Lai, Sun, Gao, Lang & Boning, "Nominality Score Conditioned Time Series Anomaly Detection by Point/Sequential Reconstruction" — **NeurIPS 2023 poster, OpenReview forum ljgM3vNqfQ** (OpenReview 직접 확인; R26 truth [B9]와 정합) | VERIFIED (2채널 2026-06-11) | 필수 | 실험 섹션 전용 |
| C-065 | §4.1.4 SOTA New | TimesNet (Wu et al., ICLR 2023) | baseline 출처 (R26 truth) | Wu, Hu, Liu, Zhou, Wang & Long — **ICLR 2023 poster, OpenReview forum ju_Uqw384Oq** (OpenReview 직접 확인; ICLR는 DOI 없음. arXiv 2210.02186 추정은 미확인 — verifier 보강) | VERIFIED (2채널 2026-06-11) | 필수 | 실험 섹션 전용 |
| C-066 | §4.1.4 SOTA New | DCdetector (Yang et al., KDD 2023) | baseline 출처 (R26 truth) | Yang et al. (KDD 2023) — DOI 10.1145/3580305.3599295, arXiv 2306.10347; VENUE_AND_PAPER_LIST Paper 2 검증 완료 | VERIFIED (2채널 2026-06-11) | 필수 | 실험 섹션 전용 |
| C-067 | §4.1.4 SOTA New | MEMTO (Song et al., NeurIPS 2023) | baseline 출처 (R26 truth) | Song, Kim, Oh & Cho (NeurIPS 2023) — VENUE_AND_PAPER_LIST Paper 13 검증 완료 (papers.nips.cc 직접 확인) + **arXiv 2312.02530 (scout 직접 확인, 2026-06-11)** + OpenReview UFW67uduJd (R26 truth [B12]) | VERIFIED (2채널 2026-06-11) | 필수 | 실험 섹션 전용 |
| C-068 | §4.1.4 SOTA New | ModernTCN (Luo & Wang, ICLR 2024 Spotlight) | baseline 출처 (R26 truth) | Luo & Wang (ICLR 2024 **Spotlight — OpenReview forum vpJMJerXHU 직접 확인 (scout 2026-06-11)**). **arXiv 버전 미발견 — OpenReview가 유일 공식본** (ICLR는 DOI 없음) | VERIFIED (2채널 2026-06-11) | 필수 | 실험 섹션 전용 |
| C-069 | §4.1.4 SOTA New | CATCH (Wu et al., ICLR 2025) | baseline 출처 (R26 truth) | Wu et al. (ICLR 2025) — arXiv 2410.12261; VENUE_AND_PAPER_LIST Paper 5 검증 완료 | VERIFIED (2채널 2026-06-11) | 필수 | 실험 섹션 전용 |
| C-070 | §4.1.4 Weak-sup | DeepMIL (Sultani et al., CVPR 2018) | baseline 출처 (R26 truth) | C-023과 동일 — Sultani, Chen & Shah, CVPR 2018, pp.6479–6488, DOI 10.1109/CVPR.2018.00678 (DBLP 확인), arXiv 1801.04264 | VERIFIED (2채널 2026-06-11) | 필수 | 실험 섹션 전용 |
| C-071 | §4.1.4 Weak-sup | WETAS (Lee et al., ICCV 2021) | baseline 출처 (R26 truth) | C-023과 동일 — Lee, Yu, Ju & Yu, **ICCV 2021** (ICML 추정은 오류), arXiv 2108.06816 확인, DOI 10.1109/ICCV48922.2021.00726 (R26 truth [B16]; verifier 재확인) | VERIFIED (2채널 2026-06-11) | 필수 | 실험 섹션 전용 |
| C-072 | §4.1.4 Weak-sup | TreeMIL (Liu et al., ICASSP 2024) | baseline 출처 (R26 truth) | C-023과 동일 — Liu, He, Liu & Li, **IEEE ICASSP 2024** (ICML/NeurIPS 추정은 오류), arXiv 2401.11235 확인, DOI 10.1109/ICASSP48485.2024.10447536 (R26 truth [B17] + ieeexplore 문서번호 일치) | VERIFIED (2채널 2026-06-11) | 필수 | 실험 섹션 전용 |
| C-073 | §4.1.4 Weak-sup | NRdetector (Wang et al., KDD 2025) | baseline 출처 (R26 truth) | Wang et al. (KDD 2025) — DOI 10.1145/3690624.3709257, arXiv 2501.11959; NRDETECTOR_DOSSIER 전수 검증 완료 | VERIFIED (2채널 2026-06-11) | 필수 | 실험 섹션 전용 |
| C-074 | §4.1.4 Q3 조건 | Q3(normalonly)가 비지도 baseline에게 가장 유리한 조건 = "같은 라벨을 각자 패러다임에서 최선으로 쓴 비교" | 실험 설계 정당화 | NRdetector §5.1 semi-supervised 변형("trained by using only normal segments") 논리 병행 | VERIFIED (2채널 2026-06-11) | 권장 | 실험 섹션 전용 |
| C-075 | §4.1.4 주석 | TSB-AD ("Elephant in the Room") — benchmark realism 문헌 인용 후보 (§4.1.1 선례 보강용) | benchmark 비판 (R29 + §16 RT MINOR-01) | Liu & Paparrizos (NeurIPS 2024 Datasets & Benchmarks Track) "The Elephant in the Room: Towards A Reliable Time-Series Anomaly Detection Benchmark" — **proceedings.neurips.cc hash c3f3c690b7a99fba16d0efd35cb83b2c 직접 확인** + OpenReview R6kJtWsTGy; abstract에서 VUS-PR 최신뢰 지표 권고 명시 (C-048 RF-008 보강 근거 확정) | VERIFIED (2채널 2026-06-11) | 권장 | 실험 섹션 전용 |

#### §4.3 Ablation Study

| ID | 위치 | 주장 요약 | 근거 유형 | 후보 reference | 상태 | 인용 강도 | R19 분류 |
|----|------|-----------|-----------|---------------|------|----------|---------|
| C-076 | §4.3 w/o GRL 행 | GRL의 능동 억제가 anomaly-OD 제외만으로 달성 불가능한 효과를 제공 | ablation 근거 (실험 결과 — TSMAE 자체) | 실험 결과 — 외부 인용 불필요. 단, GRL 원류 인용(C-036) 동반 | — | — | 실험 섹션 |
| C-077 | §4.3 Symmetric decoder 행 | 비대칭 capacity (Teacher 3L vs Student 2L)가 discrepancy 신호의 신뢰성 기여 | ablation 근거 (contribution bullet 3 load-bearing) | 실험 결과 — TSMAE 자체. SDMAE의 비대칭 근거는 간접 지지(ANCHOR_SDMAE_DOSSIER §6.1) | — | — | 실험 섹션 |

#### §4.4 Label Sparsity Analysis

| ID | 위치 | 주장 요약 | 근거 유형 | 후보 reference | 상태 | 인용 강도 | R19 분류 |
|----|------|-----------|-----------|---------------|------|----------|---------|
| C-078 | §4.4 | NRdetector의 label-noise sweep(Table 4)과 축의 의미가 다름(라벨 희소율 vs 세그먼트 노이즈율) | 선행 sweep 설계 선례 | NRdetector (Wang et al., KDD 2025) §5.3 — NRDETECTOR_DOSSIER §3.2 | VERIFIED (2채널 2026-06-11) | 권장 | 실험 섹션 전용 |
| C-079 | §4.4 동기 | 실제 환경에서 labeled anomaly 비율이 낮은 일반 케이스 검증 | 도메인 사실 + 설정 일반성 | C-007과 동일 (NRdetector §1 논리 구조 차용) | VERIFIED (2채널 2026-06-11) | 권장 | 실험 섹션 전용 |

---

### S5. §5 Conclusion

| ID | 위치 | 주장 요약 | 근거 유형 | 후보 reference | 상태 | 인용 강도 | R19 분류 |
|----|------|-----------|-----------|---------------|------|----------|---------|
| C-080 | §5 한계 | leave-one-out 50×FLOPs 비용 | 자체 한계 인정 | 자체 서술 — 외부 인용 불필요 | — | — | — |
| C-081 | §5 향후 | Complementary masking 7-pass 옵션 언급 | 자체 설계 (비사용 옵션) | 자체 서술 | — | — | — |

---

## §2. 보조 주장 (Appendix 해당 — 본문 흡수 가능)

| ID | 위치 | 주장 요약 | 근거 유형 | 후보 reference | 상태 | 인용 강도 | R19 분류 |
|----|------|-----------|-----------|---------------|------|----------|---------|
| C-082 | Appendix §A.1 | DAGMM "simplified variant following TranAD repo" — GMM energy 제거 | baseline 각주 | TranAD (Tuli et al., PVLDB 15(6), 2022, DOI 10.14778/3514061.3514067 — C-058 서지 확정) + repo github.com/imperial-qore/TranAD `src/models.py::DAGMM` (R26 truth [B5b]); RESEARCH_SYNTHESIS §⑥ provenance 결정 반영 | VERIFIED (2채널 2026-06-11) | 필수 | 실험 섹션 전용 |
| C-083 | Appendix §B.4 | epoch-budget sensitivity (optional) | 실험 근거 | 자체 실험 (미실행 placeholder) | — | — | — |
| C-084 | §3.3 보강 (Appendix §B.1) | Linear patchify vs patch_cnn: linear patchify 사용 근거 | MAE 원류 참조 | MAE: He et al. (CVPR 2022) — §3.3 1문장 정당화("학습 효율과 구현 단순성 + MAE 원류 계보") | VERIFIED (2채널 2026-06-11) | 권장 | 방법론 섹션 |
| C-085 | §3.4 선행 (보강) | Pre-Norm transformer의 학습 안정성 기여 | 기법 선례 | C-039와 동일 — Xiong et al. (ICML 2020, arXiv 2002.04745 확인) | VERIFIED (2채널 2026-06-11) | 권장 | 방법론 섹션 |

---

## §3. 요약: 근거 유형별 분류 표

> [assembler r3] 아래 비고 열의 "CANDIDATE" 표기는 scout r2 시점 기록이다. 해당 reference들은 전부 2채널 검증을 통과했으며 (49/49 VERIFIED — `VERIFICATION_LEDGER.md`), 행 상태의 정본은 §1 클레임 매트릭스(VERIFIED 2채널 2026-06-11)와 §6 갱신 로그다.

| 근거 유형 | 해당 Claim IDs | 비고 (scout r2 갱신) |
|---------|--------------|------|
| 도메인 중요성·응용 survey | C-001, C-003, C-007 | CANDIDATE — Schmidl PVLDB 2022 + Blázquez-García CSUR 확보 |
| 선행 방법 계보 (비지도 TSAD 4유형) | C-004, C-012–C-015 | CANDIDATE (구성 논문 서지 전부 확정) |
| 선행 방법 한계 | C-005, C-006, C-017 | NRdetector로 보강 가능 |
| MAE 원류 (R22) | C-026, C-033, C-084 | He et al. CVPR 2022 — CANDIDATE |
| Self-distillation 계보 (R21) | C-028, C-029, C-030, C-034 | Zhang TPAMI 2022 (DOI 확보) + SDMAE CVPR 2024 — CANDIDATE |
| GRL 원류 | C-036 | Ganin et al. JMLR 17(59) — jmlr.org 확인 |
| Focal loss 원류 | C-037 | Lin et al. ICCV 2017 — arXiv 확인 |
| PU Learning 일반론 (R20) | C-019, C-020, C-021 | CANDIDATE — du Plessis NIPS14 / Kiryo NIPS17 / Elkan&Noto KDD08 / Bekker&Davis 2020 / DevNet / Deep SAD 확보 |
| NRdetector 선행 연구 (R16/R20) | C-010, C-024, C-046, C-052, C-073, C-074, C-078, C-079 | CANDIDATE (서지 확인 완료) |
| KD for AD 계열 | C-027 | CANDIDATE — Bergmann CVPR20 + Deng&Li CVPR22 |
| TFMAE baseline | C-031, C-063 | CANDIDATE — ICDE 2024 DOI·쪽수 확정 (arXiv 없음) |
| 데이터셋 출처 6개 (R26) | C-040–C-044 | SWaT/WaDi/SMD/PSM/SMAP+MSL — CANDIDATE |
| 지표 4종 제안 논문 (R29) | C-047–C-051 | Kim AAAI 2022 (쪽수 확보), Paparrizos PVLDB 2022, Huet KDD 2022 (ACM DOI 확보), Xu WWW 2018 (arXiv 확보) — CANDIDATE |
| 지표 정당화 선례 | C-052, C-075 | NRdetector + Elephant in the Room 모두 CANDIDATE |
| benchmark realism / clean-train 가정 문헌 | C-008, C-009, C-045 | CANDIDATE — Elephant (NeurIPS 2024 D&B) 확정 |
| AR threshold 관행 | C-053 | VERIFIED — **R30 보류 해제** (A1: arXiv 2110.02642 PDF §4 Implementation details에서 r-proportion threshold(δ) verbatim 확보; §6-2) |
| 최초성 반증 부재 | C-011, C-025 | **반증 후보 발견 — 재서술 권고 (§5 scout 로그)**; 검증 결과(§6-3): SLA-VAE 반증 강도 약함(A1), Xue & Yan 저자 정정·실존 확인(A2), DACAD=TKDE 2025 transfer 설정 — **D-008 스코핑 축소 유지 (보수적)** |
| baseline 22종 출처 (R26) | C-057–C-073 | 전수 CANDIDATE (서지 확정) |
| 명칭 신규성 검증 | C-032 | NOT_FOUND(선사용 없음) → 신조어 정의 권고 |
| weakly-supervised 3종 (DeepMIL/WETAS/TreeMIL) | C-023, C-070–C-072 | CANDIDATE — CVPR 2018 / ICCV 2021 / ICASSP 2024 확정 |

---

## §4. 수요 통계 및 Scout OPEN 목록 (§M)

### 수요 통계

| 구분 | 수 |
|-----|---|
| 전체 claim 지점 (C-001 ~ C-085) | 85 |
| 인용이 필요한 claim (자체 서술·실험 결과 제외) | 72 |
| 필수 인용 강도 claim | 52 |
| 권장 인용 강도 claim | 20 |
| CANDIDATE 상태 (scout r2 이후) | 78 (= 상태 보유 79행 − NOT_FOUND 1) |
| **VERIFIED (2채널) 상태 (assembler r3 이후, 2026-06-11)** | **78 (CANDIDATE 78행 전부 전환 — 49편 reference 전수 검증 통과, QUARANTINE 0)** |
| OPEN 상태 | 0 |
| NOT_FOUND (신조어 정의 권고 — C-032) | 1 |
| (scout r2 처리 내역: 실측 OPEN 31행 → CANDIDATE 30 + NOT_FOUND 1. 종전 frontmatter "OPEN 29"는 중복 수요(C-031/C-063, C-054/C-055)를 1건으로 센 집계로 추정. 기존 CANDIDATE 행 중 식별자 보강 11행) | — |
| R26 truth로 충당 가능 (데이터셋 + baseline 22+4) | 15 (C-040~C-044 데이터셋 5개 + C-057~C-073 baseline 17개 중 확인된 것) |

### Scout OPEN 목록 — 처리 결과 (scout r2, 2026-06-11)

1. **C-009 / C-045 [해소]** — Elephant in the Room (Liu & Paparrizos, NeurIPS 2024 D&B) 서지 확정 (proceedings.neurips.cc + OpenReview R6kJtWsTGy); VUS-PR 최신뢰 권고 abstract 명시 확인. clean-train 가정의 본문 발췌는 verifier 몫.
2. **C-011 / C-025 [중대 발견]** — 최초성 반증 후보 발견: Xue & Yan (IJCNN 2022, arXiv 2207.00705), SLA-VAE (Huang et al., WWW 2022, DOI 10.1145/3485447.3511984), 보조 DACAD (arXiv 2404.11269, venue 미확정). 재서술(범위 축소) 권고 — §5 로그 참조.
3. **C-032 [부재 확인]** — "contaminated semi-supervised" 선사용 NOT_FOUND → 신조어 정의 권고. 인접 용어(contamination-resilient/-resistant) cluster 확보.
4. **C-019 / C-020 [해소]** — du Plessis NIPS 2014 + Kiryo NIPS 2017 (비용민감) + Elkan & Noto KDD 2008 (샘플선별) + Bekker & Davis 2020 survey 채택. 주의: Dist-PU는 AAAI 2022가 아니라 **CVPR 2022** — 미채택.
5. **C-023 / C-070 / C-071 / C-072 [해소]** — DeepMIL CVPR 2018 / WETAS **ICCV 2021** / TreeMIL **ICASSP 2024** 확정 (추정 venue 2건 정정).
6. **C-054 / C-055 / C-056 [해소]** — QuoVadisTAD ICML 2024, PMLR v235:43461–43476 확정 (mlr.press 직접 확인); GCN-LSTM은 QuoVadisTAD-introduced (별도 원논문 없음, R26 truth).
7. **C-031 / C-063 [해소]** — TFMAE ICDE 2024, pp.1228–1241, DOI 10.1109/ICDE60146.2024.00099 확정; arXiv 없음.
8. **C-058 / C-059 / C-060 / C-061 [해소]** — TranAD PVLDB 15(6) DOI+arXiv / USAD KDD 2020 DOI / DAGMM ICLR 2018 OpenReview / GDN AAAI 35(5) DOI 확정.
9. **C-027 [해소]** — Bergmann et al. CVPR 2020 + Deng & Li CVPR 2022 확정 ("Wang et al. CVPR 2021" 후보는 서지 불명 — 대체).
10. **C-053 [조건부 해소]** — Anomaly Transformer (ICLR 2022) AR-threshold 관행 원류 후보 + DCdetector 추종. **R30 보류 유지** — 본문/공식 구현 발췌 확보 전 사용 금지.

---

## §5. Scout 로그 (reference-scout r2, 2026-06-11)

### 5.1 특별 수요 ② — 최초성 반증 검색 결과 (중대)

**결론: 현 서술("labeled anomaly를 표현 학습의 기울기에 직접 통합하는 최초의 end-to-end 단일 다변량 TSAD 모델") 그대로는 반증 위험 — 범위 축소 재서술 권고.**

발견된 반증 후보 (전부 실존 페이지 확인):
- **Xue & Yan, "Multivariate Time Series Anomaly Detection with Few Positive Samples" (IJCNN 2022, arXiv 2207.00705)** — MTS, 소수 labeled anomaly, autoregressive 표현 학습 + "loss components to encourage representations that separate normal versus few positive examples" → 라벨이 표현 학습 loss에 직접 개입하는 end-to-end MTSAD로 읽힘. 가장 강한 반증 후보. verifier 정독 필수 (pretext가 self-supervised masked-reconstruction이 아닌 점이 유일한 잔여 차별점일 수 있음).
- **Huang, Chen & Li, SLA-VAE (WWW 2022, pp.1797–1806, DOI 10.1145/3485447.3511984)** — multivariate KPI, semi-supervised VAE + active learning. 라벨이 VAE 학습에 개입. 차별점: active-learning 루프 의존, 표현 형성 방식의 차이 — verifier 확인 필요.
- **DACAD (Darban et al., arXiv 2404.11269, venue 미확정)** — domain adaptation: source-domain labeled anomaly + contrastive 표현 학습 MTSAD. transfer 설정이므로 직접 반증은 아니나 인접.
- 비시계열 일반 AD의 라벨-표현 통합 선례: **Deep SAD (Ruff et al., ICLR 2020)**, **DevNet (Pang et al., KDD 2019)** — "first" 서술 시 도메인 한정 필수.
- GRL 메커니즘 자체의 AD 선례: **AEGR (Soft Computing 2021)** — 비지도 network AD의 gradient reversal; domain-adversarial 계열 다수 → "GRL을 AD에 처음 도입" 류 서술 금지.

**권고 재서술 골격**: "to our knowledge, the first to integrate labeled anomalies *adversarially (via gradient reversal)* into the gradients of *masked-reconstruction self-distillation* representation learning in a single end-to-end multivariate TSAD model" + Xue&Yan/SLA-VAE/Deep SAD/DevNet 차별화 인용.

### 5.2 특별 수요 ③ — "contaminated semi-supervised" 용어 선사용 검색

검색: `"contaminated semi-supervised" anomaly detection`, `"contaminated semi-supervised" time series` (2026-06-11). **고정 설정 명칭으로의 선사용 미발견 → 신조어로 정의 가능.** 인접 용어 (혼동 방지 각주 후보): RoSAS "contamination-resilient" (Inf. Process. Manag. 2023, arXiv 2307.13239), HSCL "contamination-resistant" (arXiv 2207.11789, 이미지), Takahashi et al. "Deep PU AD for Contaminated Unlabeled Data" (OpenReview Wt6K1uoMPQ — **ICLR 2026 심사 중, 미게재 → 인용 부적격, 모니터링만**).

### 5.3 NOT_FOUND / 미해결 기록

- **C-032 용어 선사용**: NOT_FOUND (의도된 부재 확인 — 신조어 정의 권고).
- **TFMAE·USAD·ModernTCN arXiv**: 부재 확인 (공식본 = ICDE / KDD / OpenReview).
- **TimesNet arXiv 2210.02186, NPSR arXiv, DeepMIL IEEE DOI(직접), WETAS/TreeMIL IEEE DOI(직접), SDMAE·MAE publisher DOI, Lin ICCV DOI, DevNet ACM DOI**: 추정/2차 확인 단계 — verifier가 공식 페이지로 보강 (현 등재 식별자만으로도 인용 성립).
- **C-053 AR threshold**: Anomaly Transformer 본문 직접 발췌 미확보 (abstract에 없음) — R30 보류 유지.

### 5.4 부수 발견 (인용 수요 외 — 팀 보고 사항)

- **모델명 충돌**: "TSMAE"는 기존 논문 존재 — Gao et al., "TSMAE: A Novel Anomaly Detection Approach for Internet of Things Time Series Data Using Memory-Augmented Autoencoder", IEEE Trans. Netw. Sci. Eng., 2022 (DOI 10.1109/TNSE.2022.3163144, ieeexplore 문서 9744555). **논문 제출 전 모델명 재고 또는 명시적 구분 필요.**
- Exathlon (Jacob et al., PVLDB 14(11):2613–2626, DOI 10.14778/3476249.3476307 — R26 truth [D5])은 실험에 사용되나 현재 claim 행 부재 — Table 1 작성 시 행 추가 필요.

---

---

## §6. Assembler 갱신 로그 (2026-06-11 — 2채널 검증 반영)

> 근거: VERIFICATION_LEDGER_A1/A2 (card↔공식소스), VERIFICATION_LEDGER_B1/B2 + refs_B1/B2.bib (blind export), P4_DIFF_REPORT (기계 diff). 통합 판정: `VERIFICATION_LEDGER.md`. 정본 서지: `refs.bib`.

### 6-1. 상태 전환

- CANDIDATE 78행 → **VERIFIED (2채널 2026-06-11)** 전환 완료. 49편 reference 전수 검증 통과 (QUARANTINE 0).
- C-032: **NOT_FOUND 유지** — 선사용 부재는 의도된 확인 결과 (신조어 정의 권고 불변). 각주용 인접 용어 2편(xu2023rosas, wang2022hscl)은 검증 통과.
- 서지 변경 반영 주의 3건: blazquez2021review **year=2022** (인쇄판 채택 — diff ①), darban2024dacad **TKDE 2025 본판** (37(8):4485–4496 — diff ②; C-011/025 셀의 "venue 미확정" 기술은 해소됨), wang2025nrdetector **pages 1551–1562** (diff ⑥).
- 저자 CRITICAL 정정 반영 4건 (인용 시 refs.bib 정본 사용): xu2018kpivae 13인, liu2024treemil(Shizhong Li), xu2023rosas(Ning Liu), xue2022fewpositive(**Feng Xue, Weizhong Yan**) + lai2023npsr(Jeffrey H. Lang — diff ③).

### 6-2. 발췌 확보로 해소된 보류 (A 검증 — A1 ledger §5 / A2 ledger 대조)

| Claim | 보류 내용 | 해소 근거 |
|-------|----------|----------|
| **C-053** | AR-threshold 본문 발췌 확보 전 사용 금지 (R30) | **보류 해제** — A1이 arXiv 2110.02642 PDF §4 Implementation details에서 "threshold δ is determined to make r proportion data of the validation dataset labeled as anomalies" verbatim 확보 |
| C-036 | GRL 수식·schedule 원문 확인 | A1 — JMLR PDF §4.2 GRL pseudo-function(Eq.16-17) + §5.2 λ_p=2/(1+exp(−γp))−1, γ=10 verbatim 확보 |
| C-037 | focal loss p_t 정의 차이 서술 근거 | A1 — arXiv 1708.02002 PDF §3.1(Eq.2) p_t 정의 + §3.2(Eq.4) FL 수식 verbatim 확보 |
| C-051 | PA 프로토콜 원전 발췌 | A1 — arXiv 1802.03903 PDF §4.2 PA 프로토콜 verbatim 확보 (단 arXiv 판 기준 — ACM 게재본 대조 잔존, 발췌 verbatim 활용은 격리 규칙 적용: VERIFICATION_LEDGER.md §3) |
| C-019/C-020 | PU 정의·양대 계열 분류 발췌 | A1 — arXiv 1811.04820 PDF §3.1.1 SCAR Definition 1 + §5 Two-step/biased learning 분류 verbatim 확보 |
| C-047/C-050 | PA%K 정의·PA 과대평가 근거 | A1 — AAAI OJS PDF §4 PA%K 정의 + Fig.6 caption (K=0↔F1_PA, K=100↔F1) verbatim 확보 |
| C-008/C-045 | clean-train 가정의 명시 서술 발췌 (Elephant) | **부분 해소** — 인용 서지(liu2024elephant·schmidl2022evaluation·데이터셋 원논문 5종)는 전부 VERIFIED이나 Elephant 본문에서 clean-train 명시 발췌는 미확보 → scout 권고대로 **데이터셋 원논문 실측 + EXPERIMENT_PROTOCOL_TRUTH 실측 중심 재서술 경로 유지** |

### 6-3. C-011 / C-025 최초성 — 검증 결과 반영

- **SLA-VAE (huang2022slavae)**: A1 판정 — abstract 분석 결과 "semi-supervised VAE + active learning" 조합으로, labeled anomaly가 표현 학습 gradient에 adversarial 방식으로 직접 개입하는 메커니즘 **불확인 → 반증 강도 약함** (active learning 루프 의존).
- **Xue & Yan (xue2022fewpositive)**: A2 — 저자 CRITICAL 정정 (**Feng Xue, Weizhong Yan**; 종전 "Yifan Xue, Yijie Yan"은 hallucination). 실존·설정 유사성은 abstract에서 확인 — 가장 강한 차별화 인용 대상 유지.
- **DACAD (darban2024dacad)**: TKDE 2025 본판 확정 — transfer(domain adaptation) 설정이므로 직접 반증 아님 (보조 차별화).
- **결론: D-008 스코핑 축소(재서술) 유지 — 보수적.** 반증 후보들의 강도가 약화·정밀화되었어도 "masked-reconstruction self-distillation 표현 학습의 기울기에 labeled anomaly를 adversarial(GRL)로 통합하는 최초" 수준의 좁힌 재서술 + 차별화 인용(Xue&Yan/SLA-VAE/Deep SAD/DevNet) 골격을 변경하지 않는다.

### 6-4. 잔존 사항

- EXCERPT_UNVERIFIED 잔존 3건 (zhang2022selfdistill, xu2018kpivae 게재본, ruff2020deepsad §3) — **서지 인용 가능·verbatim 금지** (2단계 격리 규칙: VERIFICATION_LEDGER.md §3).
- jacob2021exathlon: claim 행 부재 — §4.1.1 Table 1 Exathlon 행 작성 시 행 추가 필요 (scout §5.4 기재 사항 유지).
- 모델명 "TSMAE" 선사용 충돌 (Gao et al., IEEE TNSE 2022) — 제출 전 재고 필요 (scout §5.4); "CSMAD"는 충돌 없음 (A1 ledger §8, D-008).

---

> 본 문서 상태: 검증 완료 — VERIFIED (2채널) 78행 / NOT_FOUND 1행 (C-032, 신조어 정의 권고) / 인용 불필요(—) 6행. OPEN 0건. QUARANTINE 0건.
> 통합 판정표: `VERIFICATION_LEDGER.md` · diff 기록: `P4_DIFF_REPORT.md` · 정본 서지: `refs.bib` · IEEE 잠정 정리: `REFERENCES_IEEE.md` · card 색인: `REFERENCE_LIBRARY_INDEX.md`
> 통합 후보 목록(고유 논문 단위, card 등급 제안 포함): `SCOUT_CANDIDATE_LIST.md`
