# TEP #12 Notion 페이지 v2 — 콘텐츠 아키텍처 설계서

작성: 2026-06-11 · 대상 원고: `temp/0610/TEP/notion_page_final.md` (v1, 약 19,000자)
근거 데이터: `results/12_20260610_211815_tep_typegen_simple/{analysis_report.md §6, idv_hard_report.md, */per_fault_metrics.json}`

---

## 0. 설계 목표 — v1의 두 가지 결함과 처방

**결함 1 (문장 품질)**: v1은 명사구 종결·긴 괄호 삽입·em-dash 연쇄가 과다해 "유려하고 완전한 한국어 문장" 기준에 미달. → §5 문체 규칙으로 처방.

**결함 2 (지표 서사)**: v1은 micro(stream) G를 주 결과로 제시한 뒤 "판독 2"에서 사후적으로 "random도 G≠0이니 raw 비교 금지"라고 변명한다. 독자는 방금 읽은 주 결과표를 불신하게 된다. → **macro per-fault G(composition-등화)를 처음부터 주 지표로 선언**하고, random 행을 그 등화가 성공했다는 **내장 검증 장치**로 제시한다. micro는 토글/부록으로 강등. 서사가 "결과 제시 → 결함 고백"에서 "측정 설계 → 설계 검증 → 결과"로 바뀐다.

분량 목표: **17,000~18,500자** (v1보다 약간 짧게).

---

## 1. 전체 서사 골격 (reading spine)

한 줄 줄거리: *"라벨의 역할을 검증할 측정 장치(TEP type-disjoint + 등화된 G)를 만들었고, label-blind 기준선으로 그 장치를 교정했더니, seen/unseen 격차의 정체는 난이도가 아니라 train 오염이며, 라벨을 '제거'로만 쓰는 전략은 무력함이 드러났다 — 정확히 MAE GRL이 증명해야 할 지점이다."*

| 구간 | 독자의 질문 | 답하는 섹션 |
|---|---|---|
| 왜 이 실험 | 기존 벤치마크로는 왜 안 되나 | 요약 callout + §1.1 |
| 무엇을 만들었나 | 데이터·fold·조건 | §1.2–1.3 |
| 무엇을 재려 했나 | Q1/Q2/Q3 | §2 |
| 어떻게 쟀고 무엇이 나왔나 | 등화된 G, C_dmg, sweep, IDV | §3 |
| 그래서 무슨 뜻인가 | 인과 서사 | §4 → 행동 원칙 §5 |
| MAE에 무슨 의미 | 프로토콜 + 예측 | §6 → §7 |

**섹션 간 연결 규칙**: 각 섹션의 마지막 문장은 반드시 *다음 섹션이 답할 질문을 명시적으로 던진다*. 예시 브리지(작성 시 변형 가능):
- §1 끝 → "이 장치로 정확히 무엇을 재려 했는지를 먼저 분명히 해 둔다."
- §2 끝 → "세 질문에 대한 답은, 측정 조건을 등화하는 작업에서부터 시작한다."
- §3 끝 → "이상의 관찰을 하나의 인과 서사로 묶으면 다음과 같다."
- §4 끝 → "이 해석에서 MAE 본 실험이 따라야 할 행동 원칙을 추리면 다섯 가지다."
- §5 끝 → "이 원칙들을 본 실험 프로토콜에 다음과 같이 반영한다."
- §6 끝 → "이렇게 설계된 본 실험에서 두 가설이 각각 어떤 결과를 낳을지 미리 적어 둔다."

---

## 2. 섹션별 콘텐츠 플랜

### 머리 요약 callout (🎯 blue_bg) — 약 1,100자
- **답할 질문**: 이 페이지가 무엇이고 핵심 발견이 무엇인가 (30초 독해).
- **핵심 메시지**: ① MAE 핵심 주장("라벨은 유형 암기가 아니라 정상 모델 정화에 쓰인다") 검증용 TEP type-disjoint 실험의 simple baseline 5종 사전 실행 — 파이프라인 검증 + 해석 좌표계(floor·ceiling·보정 기준선) 확보. ② 핵심 발견: seen/unseen 격차는 난이도가 아니라 train 오염이 만들고, "라벨 = 인스턴스 제거" 전략은 부분 라벨에서 무력 → GRL purging의 차별화 지점.
- **수정**: GRL 괄호 주석을 한 구절로 축약("라벨된 이상이 정상 표현에 스며들지 못하게 gradient를 반전시키는 정화 메커니즘"). 메타 정보 3줄(실험 번호/설계 동결/평가 경로)은 유지.

### §1 진행한 실험 — 약 3,000자
- **답할 질문**: 왜 TEP인가, 정확히 어떤 데이터·fold·조건을 돌렸나.
- **핵심 메시지**: ① TEP만이 공인 fault 분류 체계로 type-disjoint split을 허용한다. ② 모든 구성은 사전 등록으로 run 번호까지 결정론적 고정. ③ 4-fold 회전 + 공유 test로 fold 간 비교 가능.
- **표/callout**: T1(데이터 구성, 압축판) · T2(4-fold 회전, 유지) · T3(실험 매트릭스, 유지) · ✅ 검증 게이트 PASS callout(유지하되 T1에서 옮겨 온 세부 — onset off-by-one 정정, per-run smoothing — 를 이 callout 한 줄로 흡수).
- **수정**: 1.1을 4문장 이내 완전한 서술문으로 재작성. IDV 3/9/15 제외 문단은 유지하되 "excluded-hard partition으로 별도 보고"까지만 (3단 검증 예고는 §2 Q3에 양보).

### §2 실험 의도와 목적 — 약 1,600자
- **답할 질문**: 이 사전 실험이 MAE 본 실험(조건 A/B/B0/C/D)의 무엇을 미리 결정하는가.
- **핵심 메시지**: Q1 격차의 원인 분해(난이도 vs 오염 vs artifact — label-blind 모델의 G가 출발점), Q2 "라벨=제거"의 한계 곡선(GRL이 넘어야 할 floor), Q3 "구분 불가" fault의 수준별 검증(window 모델의 구조적 기회).
- **수정**: Q1에 측정 원칙 예고 한 문장 추가 — *"다만 격차를 의미 있게 재려면 seen과 unseen을 같은 평가 조건에 놓아야 하며, 그 등화 방법이 §3의 첫 번째 주제다."* (macro 서사의 복선).
- **표/callout**: 없음 (산문 3문단).

### §3 결과 분석 — 약 6,500자 (페이지의 무게중심)

**3.1 측정 설계와 주 결과 — macro per-fault G.** 표 직전 안내 문단(4~5문장)이 이 페이지의 핵심 전환부다:
1. G = seen − unseen 정의, 음수 = seen이 더 나쁨.
2. **composition 문제 명시**: stream 전체(micro)로 재면 partition마다 positive rate가 다르다(seen 41.7~62.5% vs unseen 70.5~73.5%) → F1 계열 지표는 positive rate의 증가함수이므로 micro G는 모델과 무관한 구성 artifact를 포함한다.
3. **등화 선언**: 따라서 주 지표는 fault별로 동일 구성(각 fault 20 runs × 800 anomaly + 동일 FF 40 runs, positive rate 29.4%)에서 지표를 구한 뒤 seen/unseen 그룹으로 **macro 평균**한 G다.
4. **내장 검증 예고**: train을 보지 않는 random은 등화가 옳다면 G ≈ 0이어야 한다 — 표의 첫 행이 그 검증이다.

→ **T4′ (주 결과표, §3 표 재설계 참조)** → 판독 callout 2개:
- **판독 1 (📐 purple)**: random의 macro G는 ±0.002로 소멸(micro에서는 −0.03~−0.16) → micro 격차는 전액 구성 artifact였음이 확정되고, 등화의 필요성과 유효성이 동시에 증명된다. l2_norm/F-STEP은 micro −0.022가 macro +0.045로 **부호까지 반전**된 사례. (micro 원표 전체는 토글로.)
- **판독 2 (🔬 blue)**: 등화 후에도 살아남는 것이 진짜 오염 효과다 — pca/F-DS −0.161, sensor_range −0.18~−0.43, nn_distance/F-DS만 +0.093 양수. 결정적 대조: **ffonly(clean)에서 pca의 macro G는 4 fold 전부 |G| ≤ 0.009** → usable 17종의 순수 난이도 격차는 (충분히 강한 검출기 기준) 사실상 0이고, 격차는 오염이 만든다. 약한 검출기는 clean에서도 출렁인다(sensor_range/F-UNK −0.286: subtle fault 16/19/20이 seen에 몰림) — 한 문장으로만.

**3.2 오염 피해 C_dmg (macro 통일)** — 안내 1문장: "C_dmg = ffonly − contaminated를 같은 macro 척도로 잰다(random 행이 음성 대조)." → T5′ → 판독 3 (🧭 blue): 오염 흡수의 기하학 — PCA는 전역 흡수(family 전체 + near-variable spillover: unseen IDV11이 0.996→0.722 동반 하락), 1-NN은 국소 흡수(F-DS에서 거의 무손상, G 유일 양수와 정합), sensor_range는 메커니즘 붕괴. MAE는 전역형에 가까울 것이라는 예상까지.

**3.3 Noisy-label sweep** — 표 수치 불변(T6 유지). 판독 4 (⚠️ orange): 볼록 곡선 — 80% oracle 제거에도 F-DS PCA 0.836(clean 0.999), sensor_range는 100% 직전까지 0 수준 → 잔류 오염 소수가 피해의 대부분.

**3.4 IDV 3/9/15 3단 검증** — T7 유지(판정 셀 압축, ※ selection-bias 각주 유지). 판독 5 (💡 yellow): 난이도 4계층(단일 feature point → 다변량 상관 point → 시간-집계 전용 → 완전 비식별).

**3.5 평가 지표 calibration** — 표 없는 단락(4문장). **floor가 구성의 함수임을 두 숫자로**: full-stream random floor 0.765(positive rate 75.8%) vs per-fault random floor 0.48(positive rate 29.4%, T4′ random 행에서 이미 확인). excluded-hard에서 pca 0.79가 prc_auc 0.51과 함께 읽으면 무신호라는 사례 유지. macro 서사와 자연 결합되는 지점.

### §4 해석 — 약 1,700자
- **답할 질문**: §3의 관찰들이 어떤 단일 인과 서사로 묶이는가.
- 압축 문장 유지: *"label-blind 세계에 type-generalization 격차란 없다. 있는 것은 오염 격차다."*
- 번호 4개로 축소(v1은 5개): ① **난이도 기각·오염 채택** (v1의 1+2 병합 — clean에서 pca 17종 전부 0.97+, macro G ≈ 0 / contaminated에서 seen만 선택적 붕괴), ② **흡수의 기하학** (전역 vs 국소 → GRL 가치의 정식화: "전역적으로 흡수될 뻔한 fault 방향을 라벨로 도려내기"), ③ **부분 라벨의 본질적 한계** (라벨 가치가 '제거'에 머물면 구조적 패배 → signature의 within-type 일반화 필요), ④ **window 모델의 고유 영토** (IDV3/15).

### §5 인사이트 — 약 1,700자
- callout 5개로 축소(v1은 6개 — v1의 6번 subtle-set 동결은 §6 항목과 완전 중복이므로 §6에만 둠):
1. (1️⃣ blue) **seen/unseen 비교 규율**: composition 등화(macro per-fault) + within-fold matched control(Ĝ = G − G_ctrl) 없이는 어떤 격차도 해석 불가 — random 행이 산 증거.
2. (2️⃣ purple) **라벨의 가치는 '제거'가 아니라 '일반화 정화'** — oracle-removal 볼록 곡선이 MAE noisy-label 실험(P0-4)의 평가 축을 정의.
3. (3️⃣ orange) **PA%K 절대값의 함정** — floor는 stream 구성의 함수(0.765 vs 0.48), random 행·prc_auc 병기 의무.
4. (4️⃣ green) **IDV3/15 = point-wise 방법론의 공백 지대, IDV9 = negative control.**
5. (5️⃣ yellow) **TEP의 분별력은 contaminated 체제에 있다** — 의미 있는 비교 3축(오염 내성/라벨 회복/부분 라벨 유지) + Gate 0(supervised skyline의 unseen 붕괴 확인) 중요성.

### §6 MAE 실험에 어떻게 적용할 것인가 — 약 1,300자
- 번호 목록 6개 유지하되 **항목 2를 교체**: *"본 실험 seen/unseen 주 비교는 macro per-fault Ĝ로 수행한다(평가 조건 완전 동일) — 사전 등록 설계 §4.4(b)가 per-fault matched 분석을 co-primary로 둔 결정의 실증적 승격이며, stream micro는 보조 지표로 강등한다."* 나머지: ① 동일 stream·평가 anchor, ③ sweep 곡선 병치(P0-4, labeled 선택 규칙 공유), ④ excluded-hard 문구 정정("point-wise 비식별"로 한정, IDV3/15는 diagnostic 행 분리), ⑤ subtle-set [16,19,10,5,20] 동결 적용, ⑥ 보고 규율(random 행·prc_auc 병기, raw partition 비교 금지).

### §7 MAE 실험에서 기대되는 결과 — 약 2,100자
- H1/H2 정의 문단(2문장) → T8(셀 산문 30% 압축, **판별 근거 열의 Ĝ를 "macro per-fault Ĝ"로 명기**) → 🚦 red callout(Gate 0 리스크 + "floor는 예측이 아니라 좌표" 결어 — v1 문장 유지, 이 단락은 v1에서 가장 좋은 문장).

---

## 3. 표 재설계 (현재 8개 → 8개 + 토글 1개)

| # | v1 표 | 처리 | 비고 |
|---|---|---|---|
| T1 | 데이터 구성 | **압축** | Label·Run boundary 행을 한 행("라벨/경계 무결성")으로 병합, 괄호 세부는 PASS callout으로 이동 |
| T2 | 4-fold 회전 | 유지 | 변경 없음 |
| T3 | 실험 매트릭스 | 유지 | 셀 산문만 다듬기 |
| T4 | §3.1 주 결과표 (micro G) | **교체 → T4′ macro G** | 아래 스펙 |
| T5 | C_dmg | **교체 → T5′ macro C_dmg** | 아래 스펙 |
| T6 | sweep | 유지 | 수치 불변 |
| T7 | IDV 3단 검증 | 유지 | 판정 셀 압축 |
| T8 | H1/H2 예상 | 유지 | 셀 압축 + macro Ĝ 명기 |
| 신규 | micro G 원표 | **토글** | "stream micro G 원표 — 왜 보조로 강등했나" 제목의 접힘 블록. Notion 토글 미지원 시 페이지 말미 '부록 A' 소절 |

### T4′ 주 결과표 스펙 (macro per-fault G, pak_auc_f1)
- 행: **random (등화 검증선)** 맨 위 → pca_error → l2_norm → nn_distance → sensor_range. 열: F-STEP / F-RAND / F-DS / F-UNK / **ffonly(clean) macro G 범위**.
- 색상 규칙: |G| ≤ 0.02 green, 0.02 < |G| ≤ 0.10 orange, |G| > 0.10 red. 양수는 굵게.
- **검증된 수치** (analysis_report.md §6 + per_fault_metrics.json 재계산으로 교차 확인):

| model | F-STEP | F-RAND | F-DS | F-UNK | ffonly macro G 범위 |
|---|---|---|---|---|---|
| random | −0.000 | +0.001 | −0.002 | +0.000 | −0.002~+0.002 |
| pca_error | −0.093 | −0.025 | −0.161 | −0.127 | **−0.009~+0.008 (≈0)** |
| l2_norm | **+0.045** | −0.070 | −0.030 | −0.099 | −0.153~+0.138 |
| nn_distance | −0.034 | −0.018 | **+0.093** | −0.096 | −0.076~+0.066 |
| sensor_range | −0.371 | −0.322 | −0.428 | −0.179 | −0.286~+0.190 |

- ffonly macro G 전체값(필요 시 본문 인용용): pca {+0.008, −0.002, +0.005, −0.009} / sensor {+0.134, +0.050, +0.190, −0.286} / l2 {+0.104, −0.035, +0.138, −0.153} / nn {+0.031, +0.010, +0.066, −0.076} (fold 순서 STEP/RAND/DS/UNK).

### T5′ C_dmg 스펙 (macro 통일: ffonly macro seen − contaminated macro seen)
- 단일 규칙("partition 점수는 모두 per-fault macro")으로 표 간 일관성 확보. random 행 = 음성 대조(≈0).
- **검증된 수치** (per_fault_metrics.json에서 계산):

| model | F-STEP | F-RAND | F-DS | F-UNK |
|---|---|---|---|---|
| random | −0.001 | +0.000 | −0.001 | +0.000 |
| pca_error | 0.163 | 0.074 | 0.209 | 0.144 |
| l2_norm | 0.140 | 0.141 | 0.231 | 0.079 |
| nn_distance | 0.117 | 0.091 | **0.036** | 0.095 |
| sensor_range | 0.785 | 0.858 | 0.989 | 0.526 |

- 해석 열은 유지하되 한 구절로 압축(전역 흡수 / 표준화 통계 오염 / 국소 흡수 / 메커니즘 붕괴). v1의 핵심 주장(nn의 F-DS 최소 피해, pca의 F-DS 최대 피해)은 macro에서도 그대로 성립함을 확인했다.

### 표 운용 원칙 (전 표 공통)
모든 표는 **직전에 "무엇을 어떻게 읽는 표인지" 1문장** + **직후에 해석 callout 1개**(판독 N — 한 문장 제목 + 3~5문장). 표 셀에는 해석 산문을 넣지 않는다(명사구만). 검증용 재계산 스니펫: per_fault_metrics.json에서 fold별 seen/unseen fault 목록으로 pak_auc_f1을 macro 평균 (usable 17종, 3/9/15 제외).

---

## 4. 분량 예산과 삭감 목록

예산(자): 요약 1,100 / §1 3,000 / §2 1,600 / §3 6,500 / §4 1,700 / §5 1,700 / §6 1,300 / §7 2,100 / 푸터 300 ≈ **17,300**.

삭감 지목 (v1 대비):
1. **판독 2 callout 전체 삭제** — "random은 ruler" 변명은 §3.1 도입 + 판독 1로 흡수 (−700자).
2. §4 항목 1·2 병합 (−400자).
3. §5 인사이트 6(subtle-set) 삭제 — §6 항목 5와 중복 (−500자).
4. T1의 긴 괄호 세부(onset off-by-one, 5-box smoothing)를 PASS callout 한 줄로 (−300자).
5. 요약 callout의 GRL 괄호 주석 축약 (−150자).
6. T8·T7 셀 산문 압축 (−650자).
7. 판독 1(v1)의 sensor_range ffonly 단서 괄호문 → 한 문장으로 (−200자).
신규 추가: §3.1 등화 도입부 +900자, ffonly macro 수치 인용 +200자. **순감 약 1,800자.**

---

## 5. 문체 규칙 (피드백 "어색하고 읽기 불편" 직접 처방)

1. 본문은 전부 **완전한 서술형 문장**으로 끝낸다(명사구 종결은 표 셀·callout 제목만 허용).
2. 한 문장에 주장 하나. 40자를 넘는 괄호 삽입 금지 — 필요하면 별도 문장으로 푼다.
3. em-dash는 문단당 1회 이하. v1처럼 dash로 절을 이어 붙이지 않는다.
4. 영어 용어는 최초 등장 시 한국어 풀이를 한 번만 붙이고 이후 영어만 쓴다(contamination → 오염, 이후 혼용 금지하고 '오염'으로 통일; partition·fold·fault·purging 등은 영어 유지).
5. 숫자 인용은 산문 한 문장에 3개 이하. 그 이상은 표로.
6. callout은 "판독 N — (한 문장 결론)" 두괄식으로 시작해 근거 2~4문장.

---

## 6. 수치 검증 체크리스트 (writer 필수 대조)

| 수치 | 출처 |
|---|---|
| macro G (contaminated 20개 값) | analysis_report.md §6 표 (재계산 교차 확인 완료) |
| ffonly macro G·macro C_dmg | 본 설계서 §3 표 (per_fault_metrics.json에서 계산·검증 완료) |
| micro G 토글 원표 | analysis_report.md §2 pak_auc_f1 표 |
| positive rate (41.7~62.5 / 70.5~73.5 / 29.4 / 75.8%) | manifest.json + analysis_report.md §1·§6 |
| sweep 곡선 | analysis_report.md 및 sweep/ (v1 수치 그대로) |
| IDV L1/L2/L3 | idv_hard_report.md (v1 수치 그대로) |
| IDV11 spillover 0.996→0.722 | analysis_report.md §4 per-fault 표 |
| subtle-set [16,19,10,5,20], near/far 쌍 | analysis_report.md §4 |
