---
phase: 3
agent: rereviewer-redteam
directives: [R1, R8, R9, R10, R11]
inputs:
  - paper/99_reviews/p3_blueprint_redteam_r1.md (r1: BLOCKER 3 / MAJOR 10 / MINOR 5+V1·V3 / NOTE 3)
  - paper/99_reviews/p3_blueprint_fixlog_r2.md
  - paper/03_blueprint/PAPER_BLUEPRINT.md (r2)
  - paper/03_blueprint/PAGE_BUDGET.md (r2)
verification_basis: |
  형식 대조가 아니라 사실 재검증 수행: EXPERIMENT_PROTOCOL_TRUTH.md(r3) §①·②·④,
  RESEARCH_SYNTHESIS.md(r2) §①·②·③표A·⑨, NRDETECTOR_DOSSIER.md §3.1/D8/§6,
  271_CONFIG_TRUTH.md §II·VI. 독립 실측 2건: (1) PSM checkpoint patch_embed=(512,250)
  재측정(reviser 측정 bit-exact 재현), (2) loss.py GRL pos_count==0 skip 경로 직접 확인
  (Table 4 standard-split "라벨 경로 휴면" 주장의 코드 수준 검증 — read-only).
last_modified: 2026-06-11
verdict: PASS_WITH_CONDITIONS (BLOCKER 0 잔존 / 신규 MAJOR 1 / 신규 MINOR 4 / NOTE 2)
---

# Re-Review (red-team, r2): PAPER_BLUEPRINT.md + PAGE_BUDGET.md 개정본

## 판정 요약

**r1 BLOCKER 3건 전부 실질 해소** — 형식적 문구 추가가 아니라 공격을 실제로 막는 구조 변경임을 공격 재시뮬레이션 + 정본·코드 대조로 확인했다. r1 MAJOR 10건·MINOR 5건(+V1·V3)·NOTE 3건도 전건 마감 확인(부분 수용 2건은 사유 타당). 단, **RT-B02의 재구성(capacity-gap 논리)이 새로운 공격면 1건을 연다**: contribution bullet 3이 이제 decoder-depth(symmetric) ablation에 논증을 의탁하는데, 이 ablation은 정본상 미실행("ablation(층 수 조합) 필요", FM 제외 ablation도 "결과 없음")이며 **Phase 5 진입 조건(§0.4)과 EXPERIMENT_EXECUTION_TODO 집계 어디에도 등재되어 있지 않다** — warmup과 동일한 공격 패턴의 재발 경로. 이것이 유일한 신규 MAJOR이고, 수정은 등재 1줄 + 실험 완주 조건화로 충분하다.

---

## 1. BLOCKER 3건 실질 해소 판정

### 1.1 RT-B01 (test label 학습 사용 공격) — **실질 해소 (PASS)**

**§14 재구축 구조 확인**: 공격의 정확한 형태("train 라벨의 출처가 원본 test 앞 50%다")를 서두에서 먼저 인정한 뒤 5논거로 정면 방어 — r1이 지적한 "look-ahead 방어로 label-출처 공격을 받아치는 빗나간 방어"가 제거되었다.

**각 논거의 사실 검증 (전부 정본 실측 기반 확인)**:

| 논거 | 검증 결과 |
|------|---------|
| ① 재분할 정의 | 논리 구조 검증(아래) — 사실 의존 없음 |
| ② 원본 train 라벨 구조적 부재 | **확인**: EXPERIMENT_PROTOCOL_TRUTH §① L64("원본 train 파일(전부 정상)"), L66(SMAP/MSL `np.zeros` 명시적 0, `loaders.py:2602-2604`; PSM/SMD train 라벨 파일 부재, `loaders.py:1672-1675`/`1139-1142`), L41+FEEDBACK-6(SWaT train anomaly 1.63% = 11,757pts 전부 편입 A2-front 21개 region 유래). 실측 AR 열거(SWaT 1.63/WaDi 0.52·0.76/PSM 6.20/SMAP 0.70/MSL 1.70%)도 §② L95와 일치 |
| ③ 전 모델 동일 데이터 | **확인**: weak 4종 Q3=N/A 구조적 부적합(§③ L119), Q3 normalonly 제공 — 정합 |
| ④ 시간성·통일성 | **확인**: //2 통일, safe-cut "81채널 중 4채널(전부 MSL), max +166 steps" = §② 실측표(L80-87)와 정확히 일치. D-16 과장 금지 주의까지 정본의 r2 정정(B-1)을 정확히 계승 |
| ⑤ NRdetector 선례 | **dossier로 확인**: 7:3 segment split 원문 인용(NRDETECTOR_DOSSIER L87), "anomalies embedded within the training data" 원문(L53), D8 비교행(L174). **시간 순서 보존 미명시 → 단정 인용 금지** 주의(L188)까지 블루프린트가 그대로 반영 — 과잉 인용 위험 없음 |

**"편입분 라벨은 어쨌든 원래 test 라벨" 재공격 시뮬레이션 — ①이 순환 논리인가**: ①을 단독으로 읽으면 "이름을 바꿨으니 test가 아니다"라는 정의적(definitional) 방어로 들릴 수 있다. 그러나 r2 구조에서 ①은 단독으로 서있지 않다: (a) test-set leakage의 표준 정의는 "평가에 쓰이는 데이터의 정보가 학습에 개입"인데, 평가셋(뒤 50%)의 라벨·데이터는 학습 어디에도 개입하지 않음(①의 실질 내용), (b) 왜 그 라벨이어야만 하는가에 ②가 비순환적 실질 사유(다른 라벨 출처가 존재하지 않음)를 제공, (c) ⑤가 "원본 split 경계 보존" 가정 자체가 이 하위분야에서 유지되지 않는 선례를 제공, (d) Table 4(B-03 수정)가 "프로토콜 산물 아님"을 실증으로 보강. 잔여 reviewer 우려는 "leakage"에서 "비표준 벤치마크/문헌 수치와 비교 불가"로 이동하는데, 이는 전 baseline 동일 프로토콜 재실행(③) + 비표준 인정 문구 + 한계 인정 1문장으로 커버된다. **공격이 더 이상 leakage로 성립하지 않는다 — 실질 해소.**

잔여 다듬기 2건 (§4 신규 공격면의 R2-MIN-01/02): 논거 ②의 "**유일한** 구조적 장치" 과잉 주장, Intro Para 3 에코의 전칭 표현.

### 1.2 RT-B02 (warmup contribution) — **실질 해소 (PASS), 단 재구성이 신규 MAJOR 1건 유발**

**확인 사항**:
- §11 결정 ① bullet 3에서 "trained with teacher-only warmup" **실제 삭제** 확인(취소선 + 삭제 사유 명기). 재구성 bullet은 순수 구조 논리(deeper teacher의 안정 기준 + capacity-limited student의 선택적 모방 실패 → contaminated train에서의 discrepancy 신뢰성)로만 구성 — warmup 흔적 없음.
- 자기모순 해소 확인: §5.5 CRITICAL NOTE("contribution bullet에서도 warmup 문구 제거 완료"), §12 warmup 행("contribution 서술 금지"), §15 행("contribution bullet·독립 기여로 올리지 않음") — 3개 위치 전부 일관. r1이 지적한 drafter 혼란 요인 제거됨.
- MAJOR-07 연동: MECE 경계("라벨 신호의 주입(2) vs 신호가 발현되는 구조적 기판(3)") 명문화 — bullet 2/3 흡수 모호성 해소.

**재구성의 신규 공격면 — capacity-gap 자체의 ablation**: 프롬프트가 요구한 점검 결과: ablation 표에 decoder-depth 변형 placeholder는 **있다** (Table 3 행 7 "Symmetric decoder (Teacher 2L/Student 2L)" + Appendix §B.1 3L/2L/1L). 그러나:
1. RESEARCH_SYNTHESIS 표A(L94)는 이 ablation을 "**필요**"(미존재)로, FM 제외 ablation을 "**결과 없음**"(L98)으로 명시한다 — 행 7과 행 5는 warmup 행 6과 동급의 미실행 placeholder다.
2. 그런데 행 6(warmup)만 명시적 conditional 처리되고, **행 2–5·7의 실행은 §0.4 Phase 5 진입 조건에도, fixlog §7 EXPERIMENT_EXECUTION_TODO 집계(8항목)에도 등재되어 있지 않다** (항목 6은 "w/o GRL config 설계 조건"만 다룸 — 실행 등재가 아님).
3. bullet 3이 capacity-gap으로 재구성된 지금, 행 7은 bullet 3의 유일한 정량 근거(load-bearing)다. 미실행 상태로 Phase 5에 진입하면 reviewer가 "asymmetric decoder가 기여라면서 symmetric 비교가 없다"고 공격 — **warmup 공격의 정확한 재발 패턴**.

→ **R2-MAJ-01** (§4 참조). B-02 자체는 해소이나, 이 등재 누락을 닫지 않으면 같은 클래스의 공격이 bullet 3으로 자리만 옮긴다.

### 1.3 RT-B03 (방법론 vs 프로토콜 효과 분리) — **실질 해소 (PASS)**

**설계 정합성 검증**:
- **평가 test 통일**: 두 조건(standard/contaminated) 모두 동일한 원본 test 뒤 50%에서 평가 — train 구성 차이만 분리하는 올바른 통제 설계. standard 조건의 학습 데이터(원본 train)는 평가셋과 분리 유지 — leakage 없음. 정합.
- **"standard 조건에서 라벨 경로 휴면 = 사실상 비지도 모드" 주장의 기술 검증 (본 리뷰 직접 수행)**: 라벨 0인 train에서 ① force_mask_anomaly: priority 전부 0 → 무작위 마스킹으로 자연 퇴화 ✓ ② OD 분기: 전 패치 정상 → 전 masked 패치 OD(=정상 전용과 동일) ✓ ③ **GRL: `loss.py` L293-302가 `_pos_count == 0 → skip GRL loss`(`grl_cls_loss_tensor=None`)로 명시 단락** — batch 단위 positive 부재 시 GRL 손실 자체가 계산되지 않음 ✓. 즉 **동일 config로 돌려도 세 라벨 경로가 코드 수준에서 자가 비활성화**되며, 주장은 사실이다. (단, 블루프린트는 이 주장을 코드 근거 없이 단언 — R2-MIN-03에서 인용 추가 권고. 특히 실험자가 use_grl=False로 끄는 선택을 하면 §6.7이 경고한 dead-component dynamic margin 재활성화 함정에 빠지므로, "standard 조건은 **동일 config 그대로**(use_grl=True 유지) 실행"을 명시해야 한다.)
- **2단 논증의 해석 정확성**: ①(표준 조건 경쟁력 = 방법 효과) + ②(contaminated에서 제안 방법만 추가 이득 = 라벨 활용 효과)의 difference-in-differences 구조는 성립한다. baseline의 (i)→(ii) 이득은 {prefix 정상 데이터}, 제안 방법의 이득은 {prefix 정상 + anomaly windows + 라벨}이고, 후자-전자의 귀속이 "라벨 활용"인 것은 §6.5의 양적 비대칭 인정(RT MAJOR-03) + Q1 Appendix 보완과 함께 방어 가능. main text 격상(Appendix 아님)도 r1 요구 이상의 처리.
- EXPERIMENT_EXECUTION_TODO 등재(fixlog §7 항목 3) + PAGE_BUDGET 반영(§4 3.2→3.3, §1 상쇄) 확인 — B-01·B-02와 달리 이 수정은 실험 TODO 등재까지 완결.

잔여 다듬기: Table 4의 contaminated 조건에서 baseline이 받는 데이터가 Q3임을 명시(현재 "(ii) contaminated (main protocol)"로 암묵적) — R2-MIN-03에 포함.

---

## 2. r1 MAJOR 10건·MINOR·NOTE 마감 대조

fixlog 처리표와 개정본 본문을 1:1 대조. **спot 확인은 전 건 수행** (최소 6건 요구 초과 — 아래 "확인 위치"는 전부 개정본에서 직접 확인한 위치).

| r1 ID | fixlog 처리 | 개정본 확인 위치 | 판정 |
|-------|-----------|---------------|------|
| MAJOR-01 (SDMAE 계층구분 Method 본문) | §5.6(C) 서두 1문장 | §5.6(C) 첫 항목 — 정확히 r1 권고 문장 채택, 각주는 용어계보+구조차이로 축소 | **닫힘** |
| MAJOR-02 (WETAS/TreeMIL end-to-end 차별) | §4.3 1–2문장 | §4.3 "end-to-end 차별 논리 명시" 항목 — weak label=출력 결정 수준 지도 신호, 자기지도 pretext 부재(D5) 논리. dossier 정합 | **닫힘** |
| MAJOR-03 (Q3 양적 비대칭) | §6.5 인정 1문장 | §6.5 "train 데이터 양적 비대칭 인정" — Q1+Table 4 보완 연결 | **닫힘** (단 "0.5–6.2% 수준" 표현은 R2-MIN-04) |
| MAJOR-04 (test-split epoch selection) | **부분 수용** | §6.3 공개 문구 + §15 행 + §B.4 | **닫힘** — 타당성은 §3.1 |
| MAJOR-05 (GRL vs anomaly-OD 제외 분리) | §5.6(C) 논증 + §6.7 변형 2 정의 | §5.6(C) "수동 회피 vs 능동 제거" + §6.7 행 2 정의 확정 + use_grl=False 함정 경고 + Intro bridge(§3.1 Para 3) | **닫힘** |
| MAJOR-06 (데이터셋 선택 근거) | §6.2 1문장 | §6.2 선택 근거 문장(3도메인 + 운영 스트림) + Phase 4 인용 수요 | **닫힘** |
| MAJOR-07 (bullet 2/3 경계) | bullet 3 재구성 + MECE 명문화 | §11 결정 ① MECE 검증문 — "주입 vs 기판" 경계 | **닫힘** |
| MAJOR-08 ("extend analogous" 완화) | adapt/counterpart 교체 | §4.4 옵션 C 초안 "we adapt this architectural paradigm", sibling 포지셔닝 명시 | **닫힘** |
| MAJOR-09 (상한 케이스 명시) | §5.2 3단 구조 의무화 | §5.2 ②-1/②-2/②-3 명시 — RESEARCH_SYNTHESIS §② 3단 구조(L31-45)와 일치 확인 | **닫힘** |
| MAJOR-10 (warmup placeholder 행) | **부분 수용** (conditional) | §6.7 행 6 명시적 conditional + PAGE_BUDGET §3 표 conditional 표기 | **닫힘** — 타당성은 §3.2 |
| MINOR-01 (clean-train 인용) | §6.2 + §16 등재 | 확인 | 닫힘 |
| MINOR-02 (§4.5 placeholder 조건) | §6.9 작성 금지 조건 + excl22 대표성 주의 | 확인 | 닫힘 |
| MINOR-03 (§2.1/2.2 배치 "또는" 제거) | §4.1·§4.3 "전속·절대 불가" 확정 | 확인 | 닫힘 |
| MINOR-04 (§4.2/4.3 중복) | §6.1 지침 + §6.6 + 압축전략 5 | 확인 | 닫힘 |
| MINOR-05 (TS-SDMAE) | §10.1 취소선 제거 + 결정 ⑧ 신설 | 확인 (DECISION_LOG 전사는 orchestrator 이관 — fixlog §8) | 닫힘 |
| V1 (landscape 미확인) | PAGE_BUDGET 확인 플래그 + fallback | §2 전략 1, §7, §9 확인 | 닫힘 (fallback 내용은 R2-MIN-02) |
| V3 (Table 2 열 미확정) | 2지표 고정 | BLUEPRINT §6.6 + PAGE_BUDGET §2·§3 확인 | 닫힘 |
| NOTE-01 (DAGMM DECISION_LOG) | 결정 ⑦ 신설 | 확인 | 닫힘 |
| NOTE-02 (epoch 비대칭 근거) | §6.3 + §15 통합 | 확인 — 사실관계(500/10/50, eval 5/1, ⓐⓑⓒ)는 EXPERIMENT_PROTOCOL_TRUTH §④-실행(L163-164)과 일치 | 닫힘 |
| NOTE-03 ("최초" 스코핑) | §0.1 경계 박스 + §15 행 | 확인 | 닫힘 |

추가: 리뷰 외 자체 발견(d_model=512 고정 정정)을 **독립 재실측으로 검증** — PSM checkpoint `patch_embed.weight=(512,250)` 본 리뷰 직접 재측정, reviser 측정과 bit-exact 일치. 271_CONFIG_TRUTH §II(L96 `d_model=512`)와 정합. 이 정정은 정당하며, 리뷰가 잡지 못한 정본 충돌을 verifier 역할까지 수행해 잡은 것으로 평가한다.

---

## 3. 부분 수용 2건 타당성 판정

### 3.1 RT MAJOR-04 기각 사유 (validation-split 전환 = 정본 위반) — **타당**

검증: EXPERIMENT_PROTOCOL_TRUTH §④(L151-153)는 best-epoch이 **실제로** test-split pak_auc_f1로 선정됨(MAE `evaluator.py:1363-1373` + baseline `baseline_common.py:1368` 전 모델 동일)을 명기하고, M-3/REQUEST-4(L261)는 "숨기는 선택지는 없음"을 의무화한다. r1의 1안(validation split 기반으로 서술 변경)은 존재하지 않는 validation split을 서술하는 것 = 허위 기재이므로 기각이 옳다. 채택안은 r1이 스스로 제시한 2안(전 모델 동일 + sensitivity 방어)이므로 "부분 수용"의 실질은 r1 권고 이행이다.

**잔여 위험 (NOTE)**: test-set model selection은 정직 공개로도 끝까지 공격받을 수 있는 이 논문의 최약점이다. REQUEST-4 선택지 (iii)(validation-split 선정 추가 실험)을 소형(1–2 데이터셋)으로 실행해 §B.4를 optional placeholder에서 실측 sensitivity로 격상하는 것이 rebuttal 화력을 실질적으로 바꾼다 — EXPERIMENT_EXECUTION_TODO 후보로 권고 (R2-NOTE-01).

### 3.2 RT MAJOR-10 conditional 처리 — **타당**

orchestrator 지침(행 유지)과 r1 권고(행 제거) 사이의 절충으로서: (a) 명시적 conditional(완료 시에만 본문, 미완 시 삭제+§B.1 강등/생략), (b) drafter의 미완 placeholder 본문 잔류 금지 명문화, (c) "warmup은 contribution이 아니므로 행 삭제가 논증 완결성을 훼손하지 않음" — (c)는 B-02 수정(bullet 3에서 warmup 제거) 이후에만 성립하는 명제인데 그 수정이 완료되었으므로 성립. PAGE_BUDGET §3 표에도 conditional 전파 확인. 타당.

---

## 4. 신규 공격면 점검 (r2 추가 서술 대상)

### R2-MAJ-01 — ablation 미실행 suite의 Phase 5 진입 조건 누락 (B-02 재구성의 연쇄)

bullet 3의 capacity-gap 논증이 Table 3 행 7(symmetric decoder)에 의탁하게 되었으나, 정본(RESEARCH_SYNTHESIS 표A L94 "ablation(층 수 조합) 필요", L98 "FM 제외 ablation 결과 없음")상 행 5·7은 미실행이고, **§0.4 Phase 5 진입 조건과 fixlog §7 EXPERIMENT_EXECUTION_TODO 집계 어디에도 ablation suite(행 2–5·7) 실행이 등재되어 있지 않다** (행 6 warmup만 conditional, 항목 6은 w/o GRL의 config 설계 조건만). 미실행 상태로 Phase 5 진입 시 bullet 3이 warmup과 동일 패턴("기여 주장 + ablation 부재")으로 공격당한다.

**수정 방향**: ① §0.4와 EXPERIMENT_EXECUTION_TODO에 "ablation suite (Table 3 행 2–5·7) 실행 — Phase 5 진입 전 필수(최소 행 2·7), 271 canon config 기반" 등재. ② 행 7을 행 6과 동일한 명시적 conditional로 처리하거나(미완 시 bullet 3의 "reliable signal" 주장 강도 하향 지침), 필수 실험으로 못박을 것. ③ 행 5(FM)는 §12에서 이미 "ablation 근거 필요(미존재 → REQUEST)"로 인지되어 있으므로 동일 등재.

### R2-MIN-01 — §14 논거 ②의 "유일한 구조적 장치" 과잉 주장

synthetic anomaly 주입(SDMAE 류)도 "라벨 있는 train"을 만드는 구조적 장치다 — pedantic reviewer에게 반례를 헌납하는 단어. §0.3의 라벨 출처 축(합성 pseudo vs 실제 운영)이 이미 답을 갖고 있으므로, "실제 운영 라벨의 분포를 보존하는 가장 직접적인 장치" 수준으로 완화하거나 "(synthetic injection은 실제 라벨 활용 평가가 아님)" 1구를 병기.

### R2-MIN-02 — PAGE_BUDGET fallback이 RT V3 확정을 재개방

압축 전략 1의 landscape 미지원 fallback "지표 1열(PA%K-AUC F1)로 줄이고 VUS-PR을 §A.3 이동"은 V3 수정(열 구성 2지표 **고정**)과 충돌하며, r1 V3가 지적한 "왜 이 지표만 쓰는가" 공격을 그대로 부활시킨다. fallback 우선순위를 재정렬할 것: fontsize/tabcolsep/약어 → 전략 2(Table 4의 Table 2 흡수) → 지표 1열화는 **최후 수단 + V3 재결정 필요 명기**. 분량 자체는 3.93p→3.28–3.43p로 전략 거의 전부를 요구하는 빠듯한 계획이나, fallback 사다리가 명시되어 있어 현실성은 인정(Table 4 추가는 9p 내 수용 가능 판정 — §1 0.1p 상쇄 포함 총 9.0p 유지 확인).

### R2-MIN-03 — Table 4 standard-split 조건의 실행 사양 미명기

(§1.3에서 검증한 대로 주장 자체는 참이나) ① "동일 config 그대로(use_grl=True 유지) 실행 — 라벨 경로는 코드 수준에서 자가 비활성(`loss.py` pos_count==0 skip)" 명시 필요. use_grl=False로 끄면 §6.7이 경고한 dynamic margin dead-component 함정에 빠진다 — 같은 함정 경고가 ablation에는 있고 Table 4에는 없음. ② contaminated 조건에서 baseline 데이터=Q3임을 명시. EXPERIMENT_EXECUTION_TODO 항목 3에 이 두 설계 조건을 추가.

### R2-MIN-04 — §6.5 "0.5–6.2% 수준" 범위 표기의 자기 불일치

ADV MAJ-008 수정으로 §5.2는 "SMD per-machine 확정 전 전체 상한 단정 금지"를 채택했는데, §6.5 양적 비대칭 문장이 "데이터셋별 0.5–6.2% 수준"으로 6.2% 상한을 재도입 — SMD 잔여 6 machine의 train AR이 6.2%를 넘을 가능성이 미배제 상태. "실측 완료 데이터셋 기준 0.5–6.2%; SMD 확정 대기"로 §5.2와 동일 어법 통일.

### R2-NOTE-01 — epoch 비대칭·test-selection 방어의 §B.4 의존

§15 신설 행 2건(epoch 비대칭, test-selection)의 사실관계는 전부 정본 일치 확인(§④-실행 ⓐⓑⓒ). 다만 두 방어 모두 마지막 보루가 "(옵션) §B.4 placeholder"다 — optional placeholder는 rebuttal에서 무기가 되지 않는다. REQUEST-4 (iii) 소형 실험(validation-split 선정 sensitivity, 1–2 데이터셋) + baseline epoch-budget 1점 추가(예: 대표 baseline 50 epochs 재실행)를 저비용 TODO 후보로 권고.

### R2-NOTE-02 — Intro Para 3 에코의 전칭 표현

"기존 공개 벤치마크의 원본 train split에는 labeled anomaly가 구조적으로 존재하지 않아"는 본 논문 6 데이터셋에 대해 검증된 사실이나, 전칭으로 읽히면 반례(예: Exathlon — 본 프로젝트가 보유하고도 제외한 데이터셋)가 공격 재료가 된다. "the standard MTSAD benchmarks we evaluate on" 수준으로 스코핑 (Phase 4 clean-train 문헌 검증과 연동).

---

## 5. 종합 판정

**PASS_WITH_CONDITIONS.**

- **BLOCKER 3건 전부 실질 해소** — 문구 수준 봉합이 아니라 공격 구조를 바꾸는 수정임을 확인: B-01은 라벨-출처 공격에 대한 비순환 정면 답변 체계(사실 전수 검증 통과), B-02는 contribution에서 warmup 완전 제거 + 3개 위치 일관성, B-03은 통제 설계가 정합한 main-text 보조분석(코드 수준 기술 검증 통과).
- **r1 발견 23건(B3/M10/m5+V2/N3) 전건 마감** — 부분 수용 2건 모두 사유 타당(MAJOR-04는 정본 위반 회피 + r1 대안 경로 채택; MAJOR-10은 명시적 conditional로 drafter 위험 차단).
- **조건**: R2-MAJ-01(ablation suite의 Phase 5 진입 조건 등재 — bullet 3 재구성의 필연적 후속) 1건은 Phase 5 진입 전 반드시 닫을 것. MINOR 4건·NOTE 2건은 차기 수정 라운드 또는 orchestrator 전달 사항으로 처리 가능 — Phase 3 게이트를 막을 수준이 아니다.
- 검증 중 확인한 가산점: reviser가 리뷰 외 정본 충돌(d_model dynamic)을 자체 실측으로 잡아 정정했고, 본 리뷰의 독립 재실측으로 그 정정이 옳음을 확인했다.
