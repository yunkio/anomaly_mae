---
phase: 7-reentry
agent: kbs-miniauditor
directives: [R4, A2, R17, T7]
last_modified: 2026-06-11
scope: "D-015 KBS 정합화 산문 변경분 §7-3 의무 미니 감사 — highlights 재작성 5건 + keywords 7→6 + 선언 섹션 5종"
inputs:
  diff: paper/07_latex/PROSE_DIFF_LOG.md §7
  tex: paper/07_latex/main.tex (+ main_3p_measure.tex / main_5p_measure.tex / highlights.txt)
  기준: 02_venue_study/SENTENCE_CORPUS.md · 04_references/library/ (52 cards) ·
        01_research_understanding/271_CONFIG_TRUTH.md r4 · RESEARCH_SYNTHESIS.md r3 ·
        07_latex/KBS_FORMAT_REQUIREMENTS.md
---

# P7 재진입 미니 감사 r1 — KBS 산문 변경분 (highlights 5 + keywords + 선언 5종)

## 판정 요약

| 검사 | 판정 | 비고 |
|---|---|---|
| ① ai-phrasing (신 highlights 5) | **PASS** | corpus 금지/자제 패턴 0건; 명사구 2건은 highlights 관례 내 |
| ② plagiarism (highlights + 선언) | **PASS** | 변별 n-gram 외부 일치 0건; 선언 5종 = Elsevier 표준 문구 verbatim 확인 |
| ③ method-truth (highlights 5 + keyword + Data availability) | **PASS** | 3경로·비대칭·프로토콜·5지표 전부 정본 일치; 색인성 무훼손; placeholder 무모순 |
| ④ 글자수 재검증 | **PASS** | python len() 실측 79/81/84/84/83 (전부 ≤85, 공백 포함); highlights.txt 동일 |

**종합: PASS (4/4)** — 차단 결함 0건, 비차단 NOTE 3건 (§5).

---

## ① ai-phrasing — PASS

대상: main.tex:119–123 신 highlights 5개 (3종 빌드 + highlights.txt 동일 텍스트 확인).

**어휘 (corpus 부록 B.1)**: 금지 패턴 전수 무검출 — delve / showcase / underscore /
"pivotal" / "in the realm of" / landscape / seamlessly / meticulously / holistic /
"paving the way" / unlock / harness / "a testament to" / boast / "It is worth noting" /
remarkable / vital / imperative / paramount / novel / "In conclusion" — 5개 bullet 내 0건.

**구문 (부록 B.2)**:
- em-dash 절 연결 0건. H3의 `--`(en-dash)는 "Teacher--Student" 복합 고유명 표기로 담화 dash가 아님.
- 추상 덕목 3연 병렬 없음 — H2의 3항 열거는 구체 메커니즘 명사(anomaly-priority masking /
  loss bifurcation / gradient reversal)로 corpus 양성 신호(구체물 열거)에 부합.
- 의인화 없음 — H3 "amplifies"는 corpus 직접 선례 있음 (AnomTr §3 기여문 "amplify the
  normal-abnormal distinguishability" — SENTENCE_CORPUS §3-2). abstract의
  "amplifying the Teacher--Student discrepancy signal"과 동일 동사 계열.
- 무예외 전승 주장 없음 — H5 "Competitive"는 절제형 (corpus B.3-3 양성 신호; abstract
  "achieves competitive performance"와 정합).

**85자 압축의 전보문화 여부**: H1·H3·H4는 완전한 문장. H2("Three label paths: …")와
H5("Competitive on …; graceful decay …")는 명사구·세미콜론 병치 간결체 — Elsevier
highlights의 bullet 관례(명사구 허용) 내이며 어색한 전보문 아님. 판독 자연스러움 확인.

**한계 고지**: SENTENCE_CORPUS의 10개 섹션 유형에 Highlights 표본은 없음 (§1 Abstract가
최근접). 본 판정은 corpus 금지/자제 목록 + 양성 신호 + Elsevier highlights 관례(과제
전제)를 기준으로 수행했다.

---

## ② plagiarism — PASS

### highlights (n-gram 대조)

변별 n-gram 12종을 `02_venue_study/`(corpus 105문장 + dossier 2종 + 구조 문서) 및
`04_references/library/` 52 cards 전체에 grep:

| n-gram | 외부 일치 |
|---|---|
| "contaminated semi-supervised" | 0 — 검출 4건 전부 **자기 프로젝트 문서의 자기 신조어 참조** (wang2022hscl 카드 C-032 "신조어 정의", bekker2020pusurvey "우리 설정", NRDETECTOR_DOSSIER·treemil 카드 — 타 논문 원문 아님) |
| "sparse labels" / "anomaly-priority masking" / "loss bifurcation" / "amplifies anomaly discrepancy" / "masked autoencoder amplifies" / "test prefixes" / "adds test prefixes" / "exposing labeled anomalies" / "graceful decay" / "under five metrics" | 전부 0건 |
| "label sparsity" | 0 — 검출 3건은 자기 원고 §4.4 섹션명 참조 (nrdetector·deepsad 카드의 활용 맥락 메모) |

corpus verbatim 문장 → highlights 역유입(A2 역방향) 없음. 공유 표면은 분야 일반 용어
("masked autoencoder", "gradient reversal", "anomaly detection")뿐.

### 선언 5종 (Elsevier 표준 양식 — 표준 문구 사용은 의무·관례이며 표절 아님; 지정 문구 정확성만 검사)

python 문자열 대조 (공백 정규화 후 startswith):

| 선언 | 지정 문구 일치 |
|---|---|
| ② Competing interest | **verbatim TRUE** — "The authors declare that they have no known competing financial interests or personal relationships that could have appeared to influence the work reported in this paper." + [TO BE CONFIRMED] 부가만 |
| ③ Generative AI | **verbatim TRUE** — KBS GFA §7 지정 문구("During the preparation of this work the author(s) used … full responsibility for the content of the publication.")와 일치; 후행 placeholder 노트의 정책 요지(가독성/언어 개선만 허용)도 GFA와 정합 |
| ⑤ Funding 무펀딩 권장문 (tex 주석) | **verbatim TRUE** — "This research did not receive any specific grant from funding agencies in the public, commercial, or not-for-profit sectors." |
| ① CRediT | 표준 14역할 전부 tex 주석에 등재 확인 (python 대조, line-wrap 보정 후 missing 0); 가시 텍스트 배분은 9+5 유효 부분집합 (Resources 미배정 — 허용) |
| ④ Data availability | 전문이 [X --- e.g., …] placeholder — 창작 서술 없음 (③ method-truth에서 사실 정합만 검사) |

---

## ③ method-truth — PASS

### 신 highlights 5개 vs 정본 (271_CONFIG_TRUTH r4 · RESEARCH_SYNTHESIS r3)

| # | 기술 주장 | 검증 |
|---|---|---|
| H1 | "formalize the contaminated semi-supervised MTSAD setting with sparse labels" | ✓ sec1_intro:68 기여문이 동일 주장 ("We formalize the \emph{contaminated semi-supervised} setting, in which labeled anomalies coexist with unlabeled training windows"); MTSAD 약어 sec1:10 정의. "sparse labels" = R11 설정(SYNTHESIS §②-1 대부분 unlabeled + 소수 labeled) — 설정 진술이며 상한-케이스 구현(§②-2)과 모순 없음 (본문 §4.4가 상한 명시) |
| H2 | "Three label paths: anomaly-priority masking, loss bifurcation, gradient reversal" | ✓ SYNTHESIS §②-4 라벨 사용 3지점과 1:1 — ① `force_mask_anomaly=True`(anomaly-priority masking), ② 손실 방향 분기(= loss bifurcation, sec1:74에서 "restricts the Student decoder's imitation objective to normal-patch outputs"로 본문 정의 — `grl_disable_anomaly_loss` 정본 §VIII 정합), ③ GRL 분류기 타겟. "loss bifurcation"은 본문 기존 용어 (sec1:41,56,74; sec3:220; sec5:13) — 신조어 발명 아님 |
| H3 | "asymmetric Teacher--Student masked autoencoder amplifies anomaly discrepancy" | ✓ 정본 §VIII: Teacher decoder 3층 vs Student 2층 비대칭 (sec5:15 "3-layer Teacher" 본문 일치); MAE 원리(15% mask-after-encoder); GRL suppression → anomaly에서 discrepancy 증폭 (정본 GRL 행 + abstract "amplifying the Teacher--Student discrepancy signal" 동일 주장) |
| H4 | "contaminated benchmark adds test prefixes to training, exposing labeled anomalies" | ✓ 프로토콜 정본: 원본 test 시간순 앞 50% train 편입 (SYNTHESIS §④ 공통 분할 규칙, loaders.py 근거) — 데이터셋별 prefix이므로 복수형 정확; "exposing"은 R2-M3 확정 abstract 문구; 원본 train split에 labeled anomaly 부재(sec1:17)와 정합 |
| H5 | "Competitive on [N] datasets under five metrics; graceful decay with label sparsity" | ✓ 5지표 = sec4:143–161 열거 (PA%K-AUC F1 / PA%K-AUC AUC-PR / VUS-PR / VUS-ROC / Affiliation F1) — SYNTHESIS §④ 5종 표와 정확 일치 (`pa_0_f1`은 참고 전용으로 본문도 순위 제외 명시); "Competitive"는 abstract 절제 주장과 동급 — 수치 placeholder 정책(A8/R3) 내 신규 수치 주장 없음; label sparsity sweep = §4.4 (p ∈ {1.0…0.1}, R32 3속성 한계 논리 보존) |

**"[N] datasets" placeholder 처리**: 적절 — `% PH:NUM-003` 주석이 3종 main tex의
highlights 환경 내부에 보존 (abstract NUM-001과 동일 placeholder 체계); highlights.txt에는
plain text 특성상 주석 불가하나 동일 `[N]` 토큰이 잔존해 placeholder 스캔에 걸림. NOTE-2 참조.

### keyword 제거 ("Contaminated benchmark", 7→6)

색인성 훼손 **없음**: ① 자기 프로토콜 산물의 고유명사화로 확립된 색인 개념이 아님
(KBS_FORMAT_REQUIREMENTS §10-2 권고안과 일치); ② 검색 의도는 잔존 키워드
"Anomaly detection" + "Semi-supervised learning" 조합이 포섭; ③ "contaminated"는
abstract 4회 + highlights 2회(H1·H4)로 전문 검색 회수 가능. 잔존 6개: 단일 개념·복수형
회피·American spelling (-ise/-our/-yse 변이 0건) — KBS 최대 6 충족.

### Data availability placeholder 사실 정합

"[X --- e.g., The benchmark datasets analyzed in this study are publicly available from their
original sources; code will be made available at [URL] upon acceptance.]"
— **모순 없음**: 코드 절은 abstract(main.tex:110)·sec5:31과 PH:TXT-002 3개소 동기화 주석
포함 일치, R25(git 공개 예정) 정합. 전문이 대괄호 placeholder라 현재 사실 단정 없음.
fill 시점 주의는 NOTE-3.

---

## ④ 글자수 재검증 — PASS

python `len()` 실측 (공백 포함):

| # | main.tex 소스 | 렌더 기준 (`--`→en-dash 1자) | highlights.txt | ≤85 |
|---|---|---|---|---|
| H1 | 79 | 79 | 79 | ✓ |
| H2 | 81 | 81 | 81 | ✓ |
| H3 | **85** | 84 | 84 | ✓ (소스 그대로 세어도 85 — 어느 기준이든 한도 내; diff log §7.1 기재와 일치) |
| H4 | 84 | 84 | 84 | ✓ |
| H5 | 83 | 83 | 83 | ✓ |

- **highlights.txt 동일성**: 5개 bullet 전부 main.tex item과 문자 단위 동일 — 유일 차이는
  H3 `--`(소스) → `-`(ASCII hyphen, plain text) 변환으로 diff log §7.1 기재 그대로.
  헤더 "Highlights" + bullet 5개 (KBS "3 to 5 bullet points" 충족).
- **3종 빌드 동일성**: main / main_3p_measure / main_5p_measure 모두 동일 highlights·
  keywords 6종·`\journal{Knowledge-Based Systems}`·선언 5종 블록 확인 (grep 대조).

---

## ⑤ 비차단 NOTE (조치 불요 — 기록용)

1. **NOTE-1 (용어 변이)**: H5 "graceful **decay**" vs 본문 "graceful **degradation**"
   (sec4:402, sec5:22, abstract "degrades gracefully"). degradation 형은 89자로 한도 초과라
   불가피한 압축 — 주장 동일, 차단 아님. 본문 쪽 용어가 확립형이므로 향후 highlights 재손질
   기회가 있으면 통일 검토만.
2. **NOTE-2 (2-파일 동기 의무)**: PH:NUM-003 해소 시 main.tex 3종 + **highlights.txt**
   4개소를 함께 갱신해야 함 (highlights.txt는 % 주석이 없어 placeholder 추적이 [N] 토큰
   grep에만 의존). Phase 8 placeholder 해소 체크리스트에 highlights.txt 포함 권고.
3. **NOTE-3 (Data availability fill 시)**: "publicly available from their original sources"는
   SWaT/WaDi에 대해 부정확해질 수 있음 — iTrust(SUTD) 배포는 신청(request) 기반.
   fill 시 "available from their original sources (SWaT/WaDi upon request from iTrust)" 류로
   정밀화 권고. 현재는 [X --- e.g.] placeholder라 무모순.

---

## 판정

**PASS** — ① ai-phrasing PASS / ② plagiarism PASS / ③ method-truth PASS / ④ 글자수 PASS.
D-015 산문 변경분(highlights 5 + keywords 6 + 선언 5종)은 §7-3 미니 감사 기준 통과.
차단 결함 0건; NOTE 3건은 Phase 8 핸드오프 기록 사항.
