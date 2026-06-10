---
phase: 4
agent: reference-scout
directives: [T4, R19, R26]
date: 2026-06-11
status: 후보 단계 (최종 검증 전)
authority: |
  본 목록의 모든 항목은 reference-scout가 실존 페이지(arXiv abs / OpenReview / DBLP /
  publisher 공식 페이지 / PMLR / papers.nips.cc / proceedings.neurips.cc / JMLR / vldb.org)를
  직접 열람하여 실존 확인한 '후보'다. VERIFIED 승격은 후속 2인 독립 검증 파이프라인 전속.
  card 등급(FULL/LIGHT)은 R19 기준 '제안'이며 verifier/writer가 재조정 가능.
  - FULL = 논증 뒷받침용 발췌(card) 필요
  - LIGHT = baseline·dataset 표/괄호 클러스터 인용 전용
provenance_legend: |
  [scout✓] = 2026-06-11 scout가 해당 공식 페이지 직접 열람
  [R26] = NOTION_DIGEST §II-2·II-3 truth (엄격 검증 완료 — 최종 표기 전 공식 소스 재확인 대상)
  [dossier✓] = Phase 2 dossier (ANCHOR_SDMAE/NRDETECTOR) 또는 VENUE_AND_PAPER_LIST 기확인
  [verifier-TODO] = 식별자 추정/2차 확인 단계 — verifier 보강 필요
---

# SCOUT_CANDIDATE_LIST — 고유 논문 단위 통합 후보 목록

> CLAIM_CITATION_MAP r2 (scout pass) 기준 dedupe. 고유 논문 47편 (FULL 21 + FULL-cond 1 + LIGHT 25) + 선택 2편 + 특별 보고 1건.

---

## §A. FULL 제안 — 논증 뒷받침용 발췌 카드 필요 (22편)

| Key | 제목 | Venue / 연도 | 식별자 | 커버 Claim | 확인 출처 |
|-----|------|-------------|--------|-----------|----------|
| wang2025nrdetector | Noise-Resilient Point-wise Anomaly Detection in Time Series Using Weak Segment Labels | KDD 2025 | DOI 10.1145/3690624.3709257; arXiv 2501.11959 | C-003,005,006,007,010,017,022,024,046,052,073,074,078,079 | [dossier✓] NRDETECTOR_DOSSIER 전수 |
| ristea2024sdmae | Self-Distilled Masked Auto-Encoders are Efficient Video Anomaly Detectors | CVPR 2024 | arXiv 2306.12041 (publisher DOI [verifier-TODO]) | C-029,030,034,035 | [dossier✓] ANCHOR_SDMAE_DOSSIER |
| he2022mae | Masked Autoencoders Are Scalable Vision Learners | CVPR 2022 | arXiv 2111.06377 (publisher DOI [verifier-TODO]) | C-026,033,084 | [dossier✓] VENUE list Paper 11 |
| zhang2022selfdistill | Self-Distillation: Towards Efficient and Compact Neural Networks | IEEE TPAMI 44(8):4388–4403, 2022 | DOI 10.1109/TPAMI.2021.3067100; DBLP journals/pami/ZhangBM22 | C-028,034 | [scout✓] DBLP |
| ganin2016dann | Domain-Adversarial Training of Neural Networks (Ganin et al.) | JMLR 17(59):1–35, 2016 | jmlr.org/papers/v17/15-239.html; arXiv 1505.07818 | C-036 (+C-076 동반) | [scout✓] JMLR 공식 |
| lin2017focal | Focal Loss for Dense Object Detection (Lin et al.) | ICCV 2017 | arXiv 1708.02002 (IEEE DOI 10.1109/ICCV.2017.324 [verifier-TODO]) | C-037 | [scout✓] arXiv |
| xu2022anomalytransformer | Anomaly Transformer: Time Series Anomaly Detection with Association Discrepancy (Xu et al.) | ICLR 2022 Spotlight | OpenReview LzQQ89U1qm_; arXiv 2110.02642 | C-002,004,014,017,053,057 | [scout✓] arXiv + [dossier✓] Paper 1. C-053 AR-threshold 발췌는 [verifier-TODO] (R30 보류) |
| kim2022rigorous | Towards a Rigorous Evaluation of Time-Series Anomaly Detection (Kim et al.) | AAAI 2022, 36(7):7194–7201 | DOI 10.1609/aaai.v36i7.20680 | C-047,050 | [scout✓] ojs.aaai.org 공식 (PA 과대평가·random-score 주장 abstract 확인) |
| paparrizos2022vus | Volume Under the Surface: A New Accuracy Evaluation Measure for Time-Series Anomaly Detection (Paparrizos et al.) | PVLDB 15(11), 2022 | DOI 10.14778/3551793.3551830 | C-048 | [R26]/protocol-truth §④ (scout 미재열람 — verifier 재확인) |
| huet2022affiliation | Local Evaluation of Time Series Anomaly Detection Algorithms (Huet, Navarro & Rossi) | KDD 2022, pp.635–645 | DOI 10.1145/3534678.3539339; arXiv 2206.13167; DBLP conf/kdd/HuetNR22 | C-049 | [scout✓] arXiv + DBLP |
| xu2018kpivae | Unsupervised Anomaly Detection via Variational Auto-Encoder for Seasonal KPIs in Web Applications (Xu et al.) | WWW 2018 | DOI 10.1145/3178876.3185996; arXiv 1802.03903 | C-051 | [scout✓] arXiv + truth §④ |
| liu2024elephant | The Elephant in the Room: Towards A Reliable Time-Series Anomaly Detection Benchmark (Liu & Paparrizos) | NeurIPS 2024 Datasets & Benchmarks | proceedings.neurips.cc hash c3f3c690b7a99fba16d0efd35cb83b2c; OpenReview R6kJtWsTGy | C-008,009,045,075 (+C-048 VUS-PR 권고 보강) | [scout✓] proceedings.neurips.cc (VUS-PR 권고 abstract 확인) |
| sarfraz2024quovadis | Position: Quo Vadis, Unsupervised Time Series Anomaly Detection? (Sarfraz et al.) | ICML 2024, PMLR v235:43461–43476 | proceedings.mlr.press/v235/sarfraz24a.html | C-054,055,056 | [scout✓] PMLR 공식 |
| sultani2018deepmil | Real-World Anomaly Detection in Surveillance Videos (Sultani, Chen & Shah) | CVPR 2018, pp.6479–6488 | DOI 10.1109/CVPR.2018.00678; arXiv 1801.04264; DBLP conf/cvpr/SultaniCS18 | C-023,070 | [scout✓] DBLP + arXiv |
| lee2021wetas | Weakly Supervised Temporal Anomaly Segmentation with Dynamic Time Warping (Lee, Yu, Ju & Yu) | ICCV 2021 | arXiv 2108.06816 ("accepted to ICCV 2021"); DOI 10.1109/ICCV48922.2021.00726 [R26 B16, verifier-TODO] | C-023,071 | [scout✓] arXiv (venue 'ICML 2021 추정' 정정) |
| liu2024treemil | TreeMIL: A Multi-instance Learning Framework for Time Series Anomaly Detection with Inexact Supervision (Liu, He, Liu & Li) | IEEE ICASSP 2024 | arXiv 2401.11235; DOI 10.1109/ICASSP48485.2024.10447536 [R26 B17 + ieeexplore 문서 10447536 일치] | C-023,072 | [scout✓] arXiv (venue 'ICML/NeurIPS 추정' 정정) |
| bekker2020pusurvey | Learning from Positive and Unlabeled Data: A Survey (Bekker & Davis) | Machine Learning 109:719–760, 2020 | DOI 10.1007/s10994-020-05877-5; arXiv 1811.04820 | C-019,020 | [scout✓] arXiv (journal-ref·DOI 표기 확인) |
| ruff2020deepsad | Deep Semi-Supervised Anomaly Detection (Ruff et al.) | ICLR 2020 | arXiv 1906.02694 ("Published as a conference paper at ICLR 2020") | C-021 (+C-011/025 차별화) | [scout✓] arXiv |
| schmidl2022evaluation | Anomaly Detection in Time Series: A Comprehensive Evaluation (Schmidl, Wenig & Papenbrock) | PVLDB 15(9):1779–1797, 2022 | DOI 10.14778/3538598.3538602 | C-001 (+C-009,045 보조) | [scout✓] vldb.org 공식 PDF |
| xue2022fewpositive | Multivariate Time Series Anomaly Detection with Few Positive Samples (Xue & Yan) | IJCNN 2022 | arXiv 2207.00705 (IEEE DOI [verifier-TODO]) | C-011,025 (최초성 차별화 — **반증 후보 ①**) | [scout✓] arXiv |
| huang2022slavae | A Semi-Supervised VAE Based Active Anomaly Detection Framework in Multivariate Time Series for Online Systems (SLA-VAE; Huang, Chen & Li) | WWW 2022, pp.1797–1806 | DOI 10.1145/3485447.3511984; DBLP conf/www/HuangCL22 | C-011,025 (최초성 차별화 — **반증 후보 ②**) | [scout✓] DBLP |
| darban2024dacad | DACAD: Domain Adaptation Contrastive Learning for Anomaly Detection in Multivariate Time Series (Darban et al.) | **venue 미확정** (arXiv) | arXiv 2404.11269 | C-011,025 (보조 차별화 — FULL-cond) | [scout✓] arXiv html 존재 확인 (venue 확정 전 인용 시 arXiv 표기) |

## §B. LIGHT 제안 — baseline·dataset 표 / 괄호 클러스터 전용 (25편)

| Key | 제목 | Venue / 연도 | 식별자 | 커버 Claim | 확인 출처 |
|-----|------|-------------|--------|-----------|----------|
| tuli2022tranad | TranAD: Deep Transformer Networks for Anomaly Detection in Multivariate Time Series Data | PVLDB 15(6):1201–1214, 2022 | DOI 10.14778/3514061.3514067; arXiv 2201.07284; DBLP journals/pvldb/TuliCJ22 | C-004,015,018,058,082 | [scout✓] DBLP + arXiv |
| audibert2020usad | USAD: UnSupervised Anomaly Detection on Multivariate Time Series | KDD 2020, pp.3395–3404 | DOI 10.1145/3394486.3403392; DBLP conf/kdd/AudibertMGMZ20 (arXiv 없음) | C-004,012,059 | [scout✓] DBLP |
| zong2018dagmm | Deep Autoencoding Gaussian Mixture Model for Unsupervised Anomaly Detection | ICLR 2018 | OpenReview BJJLHbb0-; DBLP conf/iclr/ZongSMCLCC18 (DOI 없음) | C-004,012,018,060 | [scout✓] DBLP |
| deng2021gdn | Graph Neural Network-Based Anomaly Detection in Multivariate Time Series (Deng & Hooi) | AAAI 2021, 35(5):4027–4035 | DOI 10.1609/aaai.v35i5.16523 | C-004,013,061 | [scout✓] ojs.aaai.org 공식 |
| su2019omnianomaly | Robust Anomaly Detection for Multivariate Time Series through Stochastic Recurrent Neural Network (Su et al.) | KDD 2019, pp.2828–2837 | DOI 10.1145/3292500.3330672 | C-004,012,042(SMD 데이터셋),062 | [R26 B7/D3] (baseline+dataset 겸용) |
| wu2023timesnet | TimesNet: Temporal 2D-Variation Modeling for General Time Series Analysis | ICLR 2023 | OpenReview ju_Uqw384Oq (arXiv 2210.02186 [verifier-TODO]) | C-004,015,065 | [scout✓] OpenReview |
| fang2024tfmae | Temporal-Frequency Masked Autoencoders for Time Series Anomaly Detection | IEEE ICDE 2024, pp.1228–1241 | DOI 10.1109/ICDE60146.2024.00099; DBLP conf/icde/FangXZ0G024 (arXiv 없음) | C-031,063 | [scout✓] DBLP + ieeexplore 문서 10597757 |
| lai2023npsr | Nominality Score Conditioned Time Series Anomaly Detection by Point/Sequential Reconstruction (Lai et al.) | NeurIPS 2023 | OpenReview ljgM3vNqfQ | C-064 | [scout✓] OpenReview |
| song2023memto | MEMTO: Memory-Guided Transformer for Multivariate Time Series Anomaly Detection (Song et al.) | NeurIPS 2023 | papers.nips.cc hash b4c898eb…; arXiv 2312.02530; OpenReview UFW67uduJd | C-004,012,067 | [dossier✓] + [scout✓] arXiv |
| luo2024moderntcn | ModernTCN: A Modern Pure Convolution Structure for General Time Series Analysis (Luo & Wang) | ICLR 2024 Spotlight | OpenReview vpJMJerXHU (arXiv 없음) | C-068 | [scout✓] OpenReview (Spotlight 확인) |
| wu2025catch | CATCH: Channel-Aware Multivariate Time Series Anomaly Detection via Frequency Patching (Wu et al.) | ICLR 2025 | arXiv 2410.12261; OpenReview m08aK3xxdJ [R26 B14] | C-002,014,016,069 | [dossier✓] VENUE list Paper 5 |
| yang2023dcdetector | DCdetector: Dual Attention Contrastive Representation Learning for Time Series Anomaly Detection (Yang et al.) | KDD 2023 | DOI 10.1145/3580305.3599295; arXiv 2306.10347 | C-004,014,016,066 (+C-053 관행 추종) | [dossier✓] VENUE list Paper 2 |
| goh2016swat | A Dataset to Support Research in the Design of Secure Water Treatment Systems (Goh et al.) | CRITIS 2016 | DOI 10.1007/978-3-319-71368-7_8 | C-040 | [R26 D1]/protocol-truth §① |
| ahmed2017wadi | WADI: A Water Distribution Testbed for Research in the Design of Secure Cyber Physical Systems (Ahmed et al.) | CySWATER 2017 | DOI 10.1145/3055366.3055375 | C-041 | [R26 D2]/protocol-truth §① |
| abdulaal2021psm | Practical Approach to Asynchronous Multivariate Time Series Anomaly Detection and Localization (Abdulaal et al.) | KDD 2021 | DOI 10.1145/3447548.3467174 | C-043 | [R26 D4]/protocol-truth §① |
| hundman2018telemanom | Detecting Spacecraft Anomalies Using LSTMs and Nonparametric Dynamic Thresholding (Hundman et al.) | KDD 2018 | DOI 10.1145/3219819.3219845 | C-044 (+C-001 응용 보강 가능) | [R26 D8]/protocol-truth §① |
| jacob2021exathlon | Exathlon: A Benchmark for Explainable Anomaly Detection over Time Series (Jacob et al.) | PVLDB 14(11):2613–2626, 2021 | DOI 10.14778/3476249.3476307 | (claim 행 부재 — Table 1 Exathlon 행 작성 시 필요) | [R26 D5] |
| duplessis2014pu | Analysis of Learning from Positive and Unlabeled Data (du Plessis, Niu & Sugiyama) | NIPS 2014 | papers.nips.cc/paper/5509; DBLP conf/nips/PlessisNS14 | C-019,020 | [scout✓] papers.nips.cc |
| kiryo2017nnpu | Positive-Unlabeled Learning with Non-Negative Risk Estimator (Kiryo et al.) | NIPS 2017 Oral | proceedings.neurips.cc hash 7cce53cf…; arXiv 1703.00593 | C-019,020 | [scout✓] proceedings.neurips.cc |
| elkan2008pu | Learning Classifiers from Only Positive and Unlabeled Data (Elkan & Noto) | KDD 2008, pp.213–220 | DOI 10.1145/1401890.1401920; DBLP conf/kdd/ElkanN08 | C-020 | [scout✓] DBLP |
| pang2019devnet | Deep Anomaly Detection with Deviation Networks (Pang, Shen & van den Hengel) | KDD 2019 | arXiv 1911.08623 (ACM DOI 10.1145/3292500.3330871 [verifier-TODO]) | C-021 (+C-011/025 차별화 보조) | [scout✓] arXiv |
| bergmann2020uninformed | Uninformed Students: Student-Teacher Anomaly Detection with Discriminative Latent Embeddings (Bergmann et al.) | CVPR 2020 | arXiv 1911.02357 (IEEE DOI [verifier-TODO]) | C-027 | [scout✓] arXiv |
| deng2022reverse | Anomaly Detection via Reverse Distillation from One-Class Embedding (Deng & Li) | CVPR 2022 | arXiv 2201.10703 (IEEE DOI [verifier-TODO]) | C-027 | [scout✓] arXiv |
| xiong2020prenorm | On Layer Normalization in the Transformer Architecture (Xiong et al.) | ICML 2020, PMLR v119 | arXiv 2002.04745 | C-039,085 | [scout✓] arXiv |
| blazquez2021review | A Review on Outlier/Anomaly Detection in Time Series Data (Blázquez-García et al.) | ACM Comput. Surv. 54(3) Art.56:1–33 (2021/2022 표기 verifier 확정) | DOI 10.1145/3444690; arXiv 2002.04236; DBLP journals/csur/Blazquez-Garcia21 | C-001 | [scout✓] DBLP + arXiv |

## §C. 선택 (LIGHT-optional, C-032 신조어 각주용 인접 용어)

| Key | 제목 | Venue / 연도 | 식별자 | 용도 | 확인 출처 |
|-----|------|-------------|--------|------|----------|
| xu2023rosas | RoSAS: Deep Semi-Supervised Anomaly Detection with Contamination-Resilient Continuous Supervision | Inf. Process. Manag. 60(5), 2023 | arXiv 2307.13239 (Elsevier DOI [verifier-TODO]) | "contamination-resilient" 용어 구분 각주 | [scout✓] arXiv 존재 + ScienceDirect 등재 확인 |
| wang2022hscl | Hierarchical Semi-Supervised Contrastive Learning for Contamination-Resistant Anomaly Detection (Wang et al.) | ECCV 2022 계열 [verifier-TODO] | arXiv 2207.11789; Springer 10.1007/978-3-031-19806-9_7 (검색 결과 — 미열람) | "contamination-resistant" 용어 구분 각주 (이미지 도메인) | [scout✓] arXiv |

## §D. 특별 보고 (인용 수요 외)

| 항목 | 내용 |
|------|------|
| **모델명 충돌** | "TSMAE" 기존 논문 존재: Gao et al., "TSMAE: A Novel Anomaly Detection Approach for Internet of Things Time Series Data Using Memory-Augmented Autoencoder", IEEE Trans. Netw. Sci. Eng., 2022, DOI 10.1109/TNSE.2022.3163144 (ieeexplore 문서 9744555 — 검색 확인). **제출 전 모델명 재고 또는 명시적 구분 필요.** |
| **인용 부적격 기록** | Takahashi et al., "Deep Positive-Unlabeled Anomaly Detection for Contaminated Unlabeled Data" — OpenReview Wt6K1uoMPQ, **ICLR 2026 심사 중(미게재)** [scout✓ OpenReview]. 인용 금지, 모니터링만. |
| **오류 정정 기록** | ① WETAS venue: ICML 2021 추정 → **ICCV 2021** ② TreeMIL venue: ICML/NeurIPS 2024 추정 → **ICASSP 2024** ③ Dist-PU: AAAI 2022 표기 → **CVPR 2022** (pp.14441–14450, DOI 10.1109/CVPR52688.2022.01406, 미채택) ④ "Wang et al. CVPR 2021" (KD-AD 후보): 서지 불명 → Deng & Li CVPR 2022로 대체. |

---

## §E. 통계

| 구분 | 수 |
|-----|---|
| 고유 논문 (핵심) | 47 (FULL 21 + FULL-cond 1 + LIGHT 25) |
| 선택 (LIGHT-optional) | 2 |
| 특별 보고 | 모델명 충돌 1 + 인용 부적격 1 + 오류 정정 4 |
| OPEN 해소 | 31행 중 30행 CANDIDATE 전환 + 1행 NOT_FOUND(신조어 권고 — C-032) |
| 최초성(C-011/C-025) | 반증 후보 2+1편 발견 — 재서술 권고 (CLAIM_CITATION_MAP §5.1) |
