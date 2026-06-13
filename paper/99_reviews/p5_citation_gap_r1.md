---
phase: 5
agent: r36-auditor
directives: [R36]
last_modified: 2026-06-11
inputs:
  - paper/05_manuscript/MANUSCRIPT_v2_draft.md (본문 + Appendix 전수)
  - paper/04_references/CLAIM_CITATION_MAP.md (C-001..C-085, VERIFIED 78 / NOT_FOUND 1)
  - paper/04_references/REFERENCE_LIBRARY_INDEX.md (가용 49 key, 44 cited / 5 unused)
  - paper/04_references/library/ (개별 card 대조: bekker2020pusurvey, xue2022fewpositive)
scope: |
  R36 전방향 citation-gap 전수 스캔 (문단 단위) + R35 과잉 인용 점검.
  Abstract/Highlights/Conclusion은 무인용 관례 적용 (신규 주장 없음 확인 후 제외).
  처리안 표기: (a)=기존 49 key 지정, (b)=신규 reference 수요, (c)=주장 완화 재서술.
---

# P5 R36 Citation-Gap Audit r1 — MANUSCRIPT_v2_draft 전수 스캔

## 0. 요약

- **발견 15건**: (a) 기존 key로 해소 11건 / (b) 신규 reference 수요 **0건** / (c) 재서술 권고 4건.
- **HIGH 4건** — 단순 인용 누락이 아니라 *인용–주장 불일치 또는 자체 모순* (G-02, G-05, G-06, G-11). Phase 6 수치 주입 전에 반드시 처리.
- R35 과잉 인용: **실질 위반 0건** (§5 참조).
- 부수: 계획되었으나 미사용된 card 5건 중 3건(darban2024dacad, xiong2020prenorm, xu2023rosas/wang2022hscl)은 본 감사 발견의 해소 수단과 일치 — 신규 수요 없이 흡수 가능.

라인 번호는 `MANUSCRIPT_v2_draft.md` 기준.

---

## 1. 발견 표 (R36 — 인용 필요/불일치)

| ID | 우선 | 위치 (line) | 문장 발췌 (요지) | 필요 근거 유형 | 처리안 |
|----|------|-------------|------------------|----------------|--------|
| G-01 | MED | §1 Para 2, L125–126 | "all four families share an implicit assumption that the training data are drawn entirely from normal operations … no architectural pathway for leveraging … labeled anomalies … the best a label-aware variant can do is exclude" | 선행 방법 한계 단정 (C-005/C-006, 필수) | **(a)** `wang2025nrdetector` — card §1 발췌("performance … constrained by the lack of prior knowledge concerning true anomalies")가 논리 지지. 문장 말미 1회 인용 추가 |
| G-02 | **HIGH** | §1 Para 3, L135 | "**The only prior work** on deep semi-supervised MTSAD we are aware of, NRdetector" | 선행 연구 존재 단정 — **자체 §2.2(L172)와 모순**: xue2022fewpositive·huang2022slavae를 "Two earlier semi-supervised variational models … in multivariate time series"로 직접 인정함 | **(c)** 재서술 필수: "the closest prior work" 또는 "the only prior work that frames deep MTSAD as a PU problem" 수준으로 한정. 인용은 현행 유지 |
| G-03 | MED | §2.1, L166 | "every family above treats the training data as predominantly or entirely normal … labeled information is either discarded or treated as noise" | 선행 방법 한계 (C-017, 필수) | **(a)** `wang2025nrdetector` (C-017 지정 card). 단락 전체 무인용 — 1회 추가 |
| G-04 | MED | §2.2, L172 | "In the time-series domain, deep representation learning informed by label signals **remains rare**." | 선행 연구 희소성 (C-022, 필수) | **(a)** `wang2025nrdetector` — "novel and practical scenario" 자인 발췌(NRDETECTOR_DOSSIER R20)가 직접 지지 |
| G-05 | **HIGH** | §2.2, L172 | (xue2022fewpositive·huang2022slavae 인용 후) "their representation learning **remains largely label-agnostic**: labels enter through auxiliary loss terms rather than shaping the gradient of the latent space" | 인용–주장 불일치: xue card 발췌 2 "loss components to **encourage representations** that separate normal versus few positive examples" — 라벨이 표현 학습 loss에 직접 개입 = "label-agnostic" 단정과 충돌. auxiliary loss term도 latent의 gradient를 형성하므로 서술 자체가 자기모순 | **(c)** 재서술: 차별화 축을 card 권고대로 이동 — ① pretext 차이 (autoregressive/VAE vs masked-reconstruction self-distillation) ② 메커니즘 차이 (direct loss 가산 vs GRL adversarial gradient). "label-agnostic" 표현 제거 |
| G-06 | **HIGH** | §2.2, L174 | "CSMAD is the first end-to-end model for multivariate TSAD that integrates labeled anomalies into **the gradient of a self-supervised representation learning objective**" | 최초성 주장 — D-008 확정 골격("masked-reconstruction **self-distillation**의 기울기에 **adversarial(GRL)**로 통합하는 최초")보다 넓음. Xue & Yan의 AR pretext도 self-supervised로 읽힐 수 있어 현 문장은 반증 노출 (§1 L140은 좁은 형태로 올바름 — 두 곳 불일치) | **(c)** L174를 §1 L140과 동일한 D-008 스코핑으로 축소 + **(a)** `darban2024dacad` (보유 FULL card, 현재 미인용)를 transfer-setting 보조 차별화로 §2.2에 추가 (CLAIM_MAP C-011/C-025 지정 위치) |
| G-07 | LOW | §3.1, L199 (또는 §1 contribution 1) | "contaminated semi-supervised" 신조어 — 정의 문장은 존재하나 **인접 용어 구분 각주 부재** | 명칭 신규성 (C-032 NOT_FOUND → 신조어 정의 + 인접 용어 각주 권고) | **(a)** LIGHT-opt 전용 card 2건 활용: 각주 1개 — "contamination-resilient" `xu2023rosas` / "contamination-resistant" `wang2022hscl`와 구분 (선택적이나 card가 이 용도로 검증·보유됨) |
| G-08 | LOW | §3.4, L228 (Table A.1 L480 동반) | "Transformer encoder of depth $n_e$ (**Pre-Layer-Normalization**, multi-head self-attention, GELU)" | 기법 귀속 (C-039/C-085, 권장) | **(a)** `xiong2020prenorm` (보유 card, 미인용) — 첫 언급에 1회. 제약 준수: "시계열 한정" 안정성 서술 금지 (현재 안정성 주장 없음 → 인용만 추가; 무인용 유지도 허용 범위) |
| G-09 | MED | §3.5, L267 | "Whereas **SDMAE's anomaly-overlook supervision** operates in the target/loss space, our GRL operates in the gradient space" | 선행 방법 특성 귀속 (C-035, 필수 — §3.5 지정 위치) | **(a)** `ristea2024sdmae` 해당 문장에 인용 추가 (§2.3 각주가 "elaborated in Section 3.5"로 위임하므로 §3.5 측에 anchor 필요) |
| G-10 | MED | §4.1.1, L326 | "A defining feature of standard MTSAD benchmarks is that their original training splits **contain no labeled anomalies by construction**" — 이 위치 무인용 | 프로토콜 사실 (C-045, 필수). 주의: CLAIM_MAP §6-2 — liu2024elephant 본문에서 clean-train 명시 발췌 **미확보** → 데이터셋 원논문 실측 중심 경로 확정됨 | **(a)** 데이터셋 원논문 클러스터 `goh2016swat, ahmed2017wadi, abdulaal2021psm, su2019omnianomaly, hundman2018telemanom` (+선택 `liu2024elephant`)를 이 문장에 부착, §A.3 label-semantics 표와 연동. §1 L130의 `liu2024elephant, schmidl2022evaluation` 단독 의존도 동일 사유로 원논문 클러스터 병기 권장 |
| G-11 | **HIGH** | §4.1.4, L376 | "Under purely unsupervised learning, the most effective use of a labeled anomaly is removal as a contaminating sample **\cite{bekker2020pusurvey}**" | 인용–주장 불일치: bekker card 주의사항 명문 — "특정 방법론 주장의 근거로 **단독 인용하지 말 것**" (PU 정의·계열 survey 전용, C-019/C-020). 이 주장(C-074)의 지정 근거는 NRdetector §5.1 ("trained by using only normal segments") | **(a)** `wang2025nrdetector`로 교체(또는 병기 후 bekker 제거); 또는 **(c)** 설계 정당화 자체 서술로 완화("we grant unsupervised baselines the excision variant, their best available use of labels") |
| G-12 | LOW | §4.4, L419 | "realistic deployments record only a fraction of events" | 도메인 사실 (C-079, 권장) | **(a)** `wang2025nrdetector` 1회 (C-007과 동일 card·동일 논리) — §1 L128에서 이미 인용된 주장 반복이므로 저강도 |
| G-13 | LOW | §A.3, L571 | "PSM and SMD … training portions are treated as normal **following the field-standard assumption**" | 문헌 관행 단정 | **(a)** 해당 데이터셋 사용 원전 `su2019omnianomaly, abdulaal2021psm` 부착, 또는 **(c)** "consistent with how these benchmarks are used in prior work \cite{…}"로 완화 |
| G-14 | MED | §A.1, L512 | "we follow the simplified re-implementation of **the TranAD repository**, in which the GMM energy term is omitted" | 구현 provenance (C-082, 필수 — TranAD 인용 동반 지정) | **(a)** `tuli2022tranad` 인용 + repo 식별(github.com/imperial-qore/TranAD)을 이 문장에 부착 (§4.1.4 L370의 클러스터 인용만으로는 provenance 문장이 무인용) |
| G-15 | LOW | §1 L134 / §3.5 L272 | "a Student repeatedly exposed to anomalous patterns during training may learn to reconstruct them accurately through an indirect route" (memorization 가설) | 가설적 메커니즘 — 자체 ablation(Row 2)으로 검증되는 자체 설계 논리 | **(c)** 현행 유지 가능 (자체 가설 + §4.3 검증 구조가 이미 명시). 선택 보강만 원하면 **(a)** `audibert2020usad` (AE의 anomaly 과잉 재구성 동기) — 신규 수요 불요 |

---

## 2. 발견별 상세 근거 (HIGH 4건)

### G-02 — §1 "the only prior work" 자체 모순
- L135: "The only prior work on deep semi-supervised MTSAD we are aware of, NRdetector"
- L172 (§2.2): "Two earlier **semi-supervised** variational models addressed label scarcity **in multivariate time series** \cite{xue2022fewpositive,huang2022slavae}"
- 둘 다 deep·semi-supervised·MTSAD이므로 §1의 "only"는 원고 내부에서 즉시 반박된다. 리뷰어가 가장 쉽게 잡는 유형. D-008의 보수적 스코핑 정신과도 충돌.

### G-05 — §2.2 "label-agnostic" 특성 단정이 인용 card와 불일치
- xue2022fewpositive card(VERIFIED_A, FULL)의 핵심 발췌: "loss components to **encourage representations** that separate normal versus few positive examples" — 라벨이 표현을 형성함을 명시. card 활용 지침도 "차별화 포인트 = pretext task(AR vs masked-reconstruction), gradient reversal vs direct loss"로 고정.
- 현 문장은 (i) 사실관계 위험(인용 논문이 반박 가능), (ii) 논리 자기모순(auxiliary loss term은 latent gradient를 형성함) 이중 결함.

### G-06 — §2.2 최초성 문장이 D-008 골격보다 넓음
- D-008/CLAIM_MAP §6-3 확정 골격: "masked-reconstruction **self-distillation** 표현 학습의 기울기에 labeled anomaly를 **adversarial(GRL)**로 통합하는 최초".
- §1 L140은 이 골격을 따르나 §2.2 L174는 "the gradient of a self-supervised representation learning objective"로 일반화 — AR pretext(self-supervised로 해석 가능)에 라벨 loss를 통합한 Xue & Yan에 노출. 두 문장을 동일 스코프로 정렬할 것. 동시에 보조 차별화로 지정된 `darban2024dacad`(transfer 설정, TKDE 2025 본판 확정)가 미인용 상태 — §2.2 포지셔닝 문단에 1문장 추가 권장.

### G-11 — §4.1.4 bekker2020pusurvey 오용
- bekker card 주의사항: "Survey 논문이므로 개별 주장보다 'PU learning 전반의 개관' 용도로 인용. **특정 방법론 주장의 근거로 단독 인용하지 말 것.**"
- "비지도 학습에서 라벨의 최선 활용 = 오염원 제거"는 PU survey가 입증하는 명제가 아님. CLAIM_MAP은 이 주장(C-074)에 NRdetector §5.1을 지정. 교체 또는 자체-정당화 완화.

---

## 3. 점검했으나 비발견(정상)으로 판정한 주요 지점

| 위치 | 판정 사유 |
|------|----------|
| Abstract / Highlights / §5 Conclusion 무인용 | 관례 (신규 주장 없음 — 본문 지지 주장만 반복) |
| §1 L121–130 클러스터 (C-001~C-004, C-007, C-008) | 지정 card로 인용 완료 |
| §2.3 전체 (C-026~C-031), §3.3/§3.4 MAE·self-distillation 계보, §3.4 ganin2016dann, §3.5 lin2017focal | 지정 위치에 인용 완료; focal-variant 구분 1문장(C-037) 이행 확인 |
| §4.1.1 Table 1 / §A.3 데이터셋 출처 5종, §4.1.2 AR-threshold (xu2022anomalytransformer — R30 해제 발췌 확보), §4.1.3 지표 5종 + PA 비판 + 원전, §4.1.4 baseline 26종 클러스터 | C-040~C-075 전부 지정대로 인용 |
| §4.4 L427 label-noise sweep 구분 (C-078) | wang2025nrdetector 인용 완료 |
| §3.6 leave-one-out, §4.2/§4.3 결과 서술, Eq.(1)–(6)/(C.1)–(C.5) | 자체 설계/실험 결과 (C-038, C-076~C-077, C-080~C-081 — 인용 불요로 합치) |
| Exathlon (jacob2021exathlon) 미인용 | v2 본문은 6 family(SWaT/WaDi/PSM/SMD/SMAP/MSL) — Exathlon 미사용이 의도된 구성. card는 잔여 보유로 무해 (Phase 6에서 Exathlon 결과를 편입할 경우에만 Table 1 행+인용 추가) |

## 4. 계획-대비 미사용 card (5/49)

| key | 지정 용도 (INDEX) | 본 감사 처리 |
|-----|-------------------|--------------|
| darban2024dacad | §1 Para 4·§2.2 보조 차별화 | G-06으로 흡수 (추가 권장) |
| xiong2020prenorm | §3.4 Pre-LN | G-08로 흡수 (권장) |
| xu2023rosas | §3.1 용어 구분 각주 | G-07로 흡수 (선택) |
| wang2022hscl | §3.1 용어 구분 각주 | G-07로 흡수 (선택) |
| jacob2021exathlon | Table 1 Exathlon 행 | 비발견 — v2 구성상 의도된 미사용 (§3 참조) |

## 5. R35 과잉 인용 점검

- 전체 인용 빈도 상위: wang2025nrdetector ×9 (최근접 경쟁자 — 각기 다른 목적: 한계·프로토콜 선례·sweep 구분·baseline), xu2022anomalytransformer ×5, kim2022rigorous ×5, su2019omnianomaly ×5 — 모두 역할 분리가 명확.
- 일반 상식 수준 서술에 붙은 불필요 인용: **0건**. 경계 사례 2건 검토 후 적정 판정 — ① L122 "labeling every anomalous time point is impractical" (도메인 단정이므로 인용 정당), ② §4.1.3 L362 multi-metric philosophy 4편 클러스터 (지표 채택 방어용 — R29 지정).
- 유일한 인용 품질 문제는 과잉이 아니라 **오적합**(G-11 bekker) — §1 표에 분류.

## 6. 통계

| 구분 | 수 |
|------|---|
| 전수 스캔 문단 (본문 §1–§5 + Appendix A–C) | 약 60 문단 + 표 11 |
| 발견 총계 | **15** |
| (a) 기존 49 key로 해소 | **11** (G-01, 03, 04, 07, 08, 09, 10, 11, 12, 13, 14) |
| (b) 신규 reference 수요 | **0** |
| (c) 재서술 권고 | **4** (G-02, 05, 06, 15; G-06은 (a) darban 추가 동반, G-11·G-13은 (c) 대안 보유) |
| 우선순위 HIGH / MED / LOW | 4 / 6 / 5 |
| R35 과잉 인용 | 0 (오적합 1건은 G-11로 계상) |
| 인용–주장 불일치 (card 대조 확인) | 2 (G-05 xue2022fewpositive, G-11 bekker2020pusurvey) + 자체 모순 1 (G-02) + 스코핑 초과 1 (G-06) |

> 후속: HIGH 4건은 Phase 5 수정 라운드에서 본문 패치 필수. (a) 11건은 전부 기존 검증된 key 지정이므로 Phase 4 미니 파이프라인 가동 불요.
