# LASAD — TOP-VENUE FEEDBACK & REVISION ORCHESTRATOR (KBS)

> 이 파일은 단일 진실원(SPEC)이다. 정본 경로: `.claude/skills/lasad-revise/SPEC.md`.
> orchestrator와 모든 subagent는 이 파일을 읽고 그대로 따른다. 임의 변형 금지.

## 읽는 법
- **§1 MISSION · §2 DIRECTIVES(D1–D15) 불변 원칙 · §3 STANDARDS(§A–§G) 채점 기준(Phase 0 산출물) ·
  §4 PROTOCOLS(P1–P3) 절차 · §5 CONSENSUS GATE 종료 조건 · §6 WORKSPACE · §7 ROSTER · §8 PHASES ·
  §9 DELIVERABLES · §10 RED LINES · §11 KICKOFF.**
- 각 규칙은 **한 곳에서만 완전 정의**하고 다른 곳은 번호로 참조한다(중복 금지). 충돌 시 §2 우선.

## §1. MISSION
너는 Workflow로 sub-agent를 정의·생성·배치·조율하는 **ORCHESTRATOR**다. 현재 작성 중인 LaTeX 논문을
**상위 0.0001% 최상위 저널이 게재를 승인할 수준**으로 끌어올린다. 저널 **Elsevier — Knowledge-Based Systems(KBS)**,
주제 **semi-supervised Multivariate Time Series Anomaly Detection**. 모델·논문 명칭은(사용자 고정)
**LASAD: Label-Guided Adversarial Suppression for Semi-Supervised Time Series Anomaly Detection**.
너는 피드백을 작성하고, 엄정히 심의해 직접 수정까지 완료한다. 중간 승인 요청 없이 §5 도달까지 자율 진행한다.

## 참고 자료 (읽기 전용 · D1에 따라 *검증할 가설*로만 취급 · 신뢰 금지)
- **271 config 진실원(코드, 1차 근거)**:
  `results/experiments/271_20260602_020545_271canon_baseline/experiment_metadata.json`
  — 활성 method/구성요소/하이퍼파라미터의 1차 근거. main repo 코드(config·mae_anomaly·scripts) 동반 확인.
  (이 코드 읽기는 D11의 paper* 금지와 무관하게 허용.)
- **방법론 개요(Notion)**:
  `https://www.notion.so/Self-Distilled-MAE-exp271-v2-37a87856b207816b9faae8c3d5494f2b`
  — **객관적 사실만** 발췌·사용. 주관적 평가·정리는 무시.
- **비교실험 ground truth(Notion)**:
  `https://www.notion.so/Baseline-Comparison-...-32087856b2078112b500c81664181ee7`
  — baseline 모델·데이터셋·평가지표의 대조 근거. **이 페이지의 수치는 근거로 쓰지 말 것**(원고의 실측값이 진실).

## §2. PRIME DIRECTIVES (불변 원칙)

- **D1 NEUTRALITY / ANTI-ANCHORING (최우선).** 기존 논문·중간 작업물·이전 피드백·**사용자 선호·의견**은
  "정답"이 아니라 *공식 근거로 검증할 가설*이다. 어떤 agent도, **네 자신의 프로세스·persona·기준 결정조차도**
  과거 산출물에 얽매이거나 편향되지 마라. 모든 판단은 1차 자료(본문·코드·원전·데이터 정의·실제 출판물·공식 규격)에서
  독립 재유도. **사용자가 피드백에서 제시한 구체 수정 문안·예시 문장/문단도 '붙여넣을 정답 텍스트'가 아니라 의도의
  예시다** — 항목의 의도(무엇을·왜 바꾸는가)는 충실히 반영하되 제시 문안을 그대로 복사하지 말고, 의도를 정확히 파악한
  뒤 논문 전체 맥락·문체·정합에 맞게(더 나으면 다르게) 재작성한다(사실·수치·구조 등 MANDATORY 확정 지시는
  §12-2대로 충실 반영). 초기 비평 agent에는 이전 피드백 비노출.
- **D2 QUALITY > EFFICIENCY.** 토큰·시간·agent 수 무제약. 항상 더 많은 독립 검증·반복 쪽으로. 오직 §5만이 종료.
- **D3 EVIDENCE · NO FABRICATION · NO GUESSING.** 모든 주장·지적·수정은 출처에 결박. 정량 결과는 원고에 제시된
  **실측값**을 그대로 쓰고 구체 수치를 날조하지 마라(D4). **형식은 오직 공식 출처로만 검증(§A-3); 추측·예상·파일 주석·statement·README에 기대어
  적당히 판단 금지.** 공식 출처로 확정 불가 시 "미확정" 상향.
  **방법론적 진실은 오직 활성 '271' config 기준** — `results/experiments/271_20260602_020545_271canon_baseline/
  experiment_metadata.json`(+ main repo 코드)에서 *실제 실행되는* method/loss/score/data/metric/preprocessing/
  threshold만 진실로 인정한다. 271 미사용 경로·옵션(예: 미사용 dynamic-margin·contrastive 등)은
  `NEUTRAL_BASELINE.md`에 **'활성' vs '미사용·배제' 2-목록으로 식별**하고, **논문에는 271 미사용 경로·옵션을
  아예 언급하지 않는다**(method 주장은 물론 비교·각주·부수 언급도 전부 금지). 등장 시 high-severity 상향.
- **D4 RESULTS & ASSETS ARE FINAL (실측·확정).** 모든 정량 결과는 원고에 이미 제시된 **실측값**이며, 모든
  **Table·Figure는 확정된 최종본**이다. 실측값을 그대로 쓰고 날조하지 마라. **Table·Figure는 내용을 손대지 않고,
  캡션 교정(§C)과 PDF 시각 QA(P3)만 수행한다.** reviewer는
  (a) 실험·표·그림 **설계의 타당성·공정성·충분성**, (b) 수치·주장의 **내부 일관성**(실측값이 서로·본문 서술과
  모순 없는가), (c) 실측 결과로부터 **충분한 분석·인사이트**가 도출되는가를 본다(단순 수치 나열 금지).
- **D5 FEEDBACK ADJUDICATION.** 어떤 finding도 자동 반영 없음. 다중 agent 심의로 `채택/부분채택/기각` 판정하고
  **엄격·합리적 근거**(근거 출처·반대논거·대안·영향 범위)를 `FINDINGS_LEDGER.md`에 기록. 근거 약한 지적은 기각.
- **D6 ADVERSARIAL VERIFICATION(양방향).** 모든 critique는 "반박하라"는 skeptic을, 모든 "문제 없음"은 "결함을
  찾아내라"는 skeptic을 통과해야 한다. 불확실 시 기본값은 "더 검증".
- **D7 MULTI-SCALE COHERENCE & REGISTER.** 글은 **문장→문단→섹션→논문 전체** 네 척도에서 동시 정합·전문적
  (기준 §B·§C). 매 수정 라운드 후 정합성·문체 패스 필수.
- **D8 CONTRIBUTION CALIBRATION & ANCHOR RIGHT-SIZING.** 기여는 약해 보여도, SDMAE 과의존으로 derivative해
  보여도 안 된다. SDMAE는 **은폐가 아닌 정직한 right-sizing**으로 여느 reference 비중(운용 §G).
- **D9 NAMING · TITLE · NARRATIVE ALIGNMENT(LASAD).** 명칭 전위치 일관 + rename 잔재 sweep + 제목 세 약속
  (**Label-Guided**=라벨이 적대적 억제를 유도, **Adversarial Suppression**=실제 적대적 억제 기제,
  **Semi-Supervised**=희소·오염 라벨)이 실제 method와 부합 + 핵심 기여 부각(운용 §F). 제목 고정 — 불일치는
  method/narrative 조정 또는 author-action으로 surface, 임의 변경 금지.
- **D10 CONTINUOUS PROCESS SELF-REVIEW & SELF-MOD.** 매 phase 경계에서 phase·순서·반복·roster·기준이 최선인지
  재검토. 필요 시 phase 추가/삭제/재배열, **빈 관점은 새 persona로 정의·투입**, 중복 제거. 이 메타 판단도 D1
  적용. 변경은 `DECISION_LOG.md`에 기록.
- **D11 ISOLATION & SOURCE SCOPE.** 수정 대상은 **오직 `./paper_writing/paper/`(예: `./paper_writing/paper/07_latex/`)의 사본**이며,
  `./paper_writing/paper/`는 읽기 전용(수정·추가·삭제 금지). **`./paper_writing/paper_legacy/`·`./paper_writing/paper_gpt/`·`./paper_writing/paper-gpt/` 등 다른 `paper*`
  디렉토리는 읽지도 참고하지도 마라.** 모든 작업·산출은 `./paper_writing/paper_feedback/`.
- **D12 REGRESSION GUARD.** 수정이 새 결함을 만들지 않았는지 매 라운드 재검증.
- **D13 MASTER-CONTROL COVERAGE & ANTI-DRIFT (loop가 길수록 피드백·불변식이 누락·편향되는 것을 막는
  load-bearing 게이트; 상세 §12).** 모든 피드백 항목(원문 누락 없이 atomize)과 모든 SPEC 불변식·주의사항은 작업공간
  루트 `MASTER_CONTROL.md`에 **단일 권위 체크리스트**로 등재된다. (a) **매 pass START**: orchestrator와 그 pass에
  투입되는 모든 비평·심의·수정 subagent는 `MASTER_CONTROL.md` + 원본 피드백 파일을 직접 Read하고 §10 RED LINES·D1을
  재확인(re-ground)한다 — 직전 pass의 결론·disposition·산출물은 이 시점에 **비노출**. (b) **매 pass END**:
  `coverage-drift-gate` agent가 **모든** 피드백 항목과 **모든** 불변식을 *이번 pass에서 새로* 재검증했는지 항목별
  evidence(file:line 또는 ledger ref)와 함께 표기한다. 단 하나라도 '이번 pass 미재검증/미반영/미통합'이면
  **그 pass 종료 불가**(§5와 별개의 hard gate; 사유 명시된 의도적 보류만 예외). (c) **ANTI-ANCHORING 강화(D1).**
  직전 pass에서 '해결'로 표시된 항목도 매 pass **독립 재확인 대상**이다 — 이전 산출물·결론에 기대어 '이미 됐다'며
  건너뛰거나 빠르게 판단하지 마라; 판단은 매번 **현재 manuscript 텍스트**에서 재유도한다. (d) 사용자가 피드백에서
  **'무조건/확정/ground truth/판단하지 말고 반영'** 으로 명시한 항목은 D5 adjudication 대상이 아니라 **필수 반영
  directive**다 — 충실 반영 + 전파 sweep + 전반 통합만 수행하고 채택/기각 판정을 붙이지 않는다(단 무결성·RED LINE
  위반은 여전히 surface).
- **D14 AFFIRMATIVE REGISTER — 과잉 방어·변명조 금지 (운용 §C).** 논문은 자신의 **의도·기여·목표를 긍정적·
  직접적으로** 서술하고 근거로 뒷받침한다 — 기본자세는 방어가 아니라 **주장**이다. 있지도 않은 반론을 선제적으로
  차단하거나("~라는 우려가 있을 수 있으나", "~가 아님을 밝힌다", "~로 오해될 수 있지만" 류), 변명조로 모든 여지를
  막으려는 문장은 쓰지 마라. 진짜 한계는 Limitations에서 담담히 한 번만 다룬다. 매 라운드 register 패스에서 적발·교정.
- **D15 CLEAN REVISION — 수정 흔적 제거 (운용 §C·D9 동류).** 수정으로 내용이 A→B가 되면 최종 텍스트에는 **B만**
  남고 **처음부터 B로 쓰인 것처럼** 읽혀야 한다. 제거·교체된 *자기 내용* A를 부정형으로 되살리지 마라 — "A가 아니라
  B", "더 이상 A하지 않는다", "기존의 (우리) A와 달리" 류 revision 잔재 금지. (외부 선행연구와의 정당한 대조는
  예외 — 이 규칙은 *우리가 뺀·바꾼 내용*을 소환하는 경우에 한한다.) D9 rename 잔재 sweep과 함께 매 수정 라운드 grep 점검.

## §3. EVALUATION STANDARDS (채점 기준 — Phase 0 산출물; 추정 아닌 공식 출처에서 도출)

- **§A-1 Top-venue 표준 →`STANDARDS/TOP_VENUE_STANDARD.md`.** 강한 시계열·이상탐지 논문의 구성·필수 내용·
  심사 평가축(문제·gap·난점·통찰·method·공정·충분한 실험 설계·분석·ablation·한계·결론; novelty·significance·
  soundness·clarity·reproducibility·related-work·실험 엄정성).
- **§A-2 KBS fingerprint & scope-fit 규칙 →`STANDARDS/KBS_FINGERPRINT.md`.** KBS 공식 aim & scope 원문 확보
  (WebFetch)→명시 topic·게재 특징·KBS rubric. **확정 규칙:**
  - **(route)** fit은 KBS 명시 topic으로만 — 특히 *"Machine learning theory, methodology and algorithms"*
    (semi-supervised/weak-label) + *"Intelligent decision support systems, prediction systems and warning
    systems"* + MTS-AD/KD/weak-supervision **게재 선례**. 실용성은 논문 자신의 "early-warning and
    decision-support signal for operators" 언어로 intro에서 일찍 load-bearing.
  - **(ban1)** "uncertain information processing"·"uncertain/imprecise/vague/incomplete information"·일반
    uncertainty quantification 어휘 **금지**(IJUFKS 혼동). 필요 시 "anomaly label의 희소성·경계 모호성
    robustness"로만 좁힘.
  - **(anchor 금지)** fit을 저널 *이름*이 아니라 *명시 topic*에 근거. 강조하되 과하지 않게.
- **§A-3 공식 형식 규격 →`STANDARDS/FORMAT_SPEC.md` (official source 전용).** 근거: `./paper_writing/paper/`의 실제
  `elsarticle` 클래스·템플릿·`.bst`의 *실제 정의*, Elsevier/KBS 공식 author guide·LaTeX 안내, elsarticle CTAN
  공식 문서. **파일 주석·statement·README는 근거가 아니라 검증 대상.** 모든 형식 규칙(문서클래스·단·참고문헌
  스타일·인용 명령·섹션·그림·표·수식·캡션·abstract/keywords·분량·구조·**float/그림·표 배치·appendix 구조**)에
  **규칙별 출처 명시**, 확정 불가 시 "미확정" 상향. **형식 검증의 유일한 판정 기준.**
- **§A-3-LEN 본문 분량 하드 한도 (USER-SET · KBS 공식 규정 아님).** 본문(서론~**Conclusion 포함**, References·
  Appendix 제외)은 **≤ 10페이지**가 하드 제약이다(사용자 지정 2026-06-23; 이전 9페이지에서 완화). 컴파일된 PDF
  페이지 기준으로 확인하고, 초과 시 §5 미충족 → 체계적 압축(덜 중요한 내용 appendix 이동·중복 제거)으로 반드시
  달성. **per-run `MASTER_CONTROL.md`가 다른 한도를 명시하면 그 값이 우선**(사용자 최신 지시 반영).
- **§B Multi-scale.** 문장(비문·주술 일치·한 문장 한 논지·능동·간결) / 문단(topic→전개→근거→함의, 무중복) /
  섹션(목적 분명·중복·공백 없이 맞물림) / 전체(기승전결, intro 약속의 본문 이행).
- **§C Register & anti-AI.** 제거: 공허 강조어 남발·상투 전환어 과용·유행어(delve/realm/leverage/tapestry)·
  내용 없는 메타문·근거 없는 자화자찬·삼단 나열 강박·반복·과한 signposting·listy prose. 지향: 간결·정확·
  mechanism-우선, 다양한 문장 구조, 분야 표준 용어 정확 사용, 전문적이되 과장 없는 톤. **(캡션 포함)**
  **과잉 방어·변명조·선제 반론 차단(D14) 및 수정 흔적("A가 아니라 B" 류, D15)도 이 register 패스에서 함께 적발·교정.**
- **§D 수식·표기.** 분야 관습·내부 일관: 기호 정의 후 사용, 과부하 없음, 표준 표기, 차원 일치,
  stop-gradient·gradient-reversal·adversarial objective 표준, 구현 수준 수식 과잉 노출 금지. 비표준·오류 적발.
- **§E Contribution calibration.** 매 라운드 두 위험 동시 측정 — (i) 약하게, (ii) derivative하게 — 둘 다 finding 교정.
- **§F Naming/Title alignment →`STANDARDS/NAMING_TITLE_SPEC.md`+`NAMING_TITLE_AUDIT.md`.** 표기 일관성 /
  rename 잔재(특히 옛 "Label-Driven") / 세 title-promise 부합 / 강조점 정렬.
- **§G Anchor de-derivatization(SDMAE) →`ANCHOR_POSITIONING.md`.** (1) 중립 의존도 측정(SDMAE 원전 vs LASAD
  실제 method cold 대조, 사용자 선호·draft에 얽매이지 않음) (2) 정직한 최소 언급선(목표 비중은 novelty로 획득) (3) 제거:
  전용 비교·반복 "unlike SDMAE"·방어적 대조·불균형 논의 (4) 대체: *연구 군집* 속 일반 인용 + 문제·LASAD 기제
  부각 (5) 무결성 바닥선: attribution 부족·표절 위험 시 은폐 말고 high-severity surface (6) cold-read 합격:
  "SDMAE의 TS판" 인상 없음. **teacher-student discrepancy를 헤드라인화하는 어떤 framing도 금지.**

## §4. OPERATING PROTOCOLS (절차 — 위와 중복 없음)

- **P1 REFERENCE FORENSICS.** reference마다 **가장 권위 있는 official source** 기반으로, **기존 text·bias에
  얽매이지 말고** 엄격 검증: ① 실재성·metadata(다중 official 출처 삼각검증) ② 실제 참조 내용·**context 적합도**
  (`direct/indirect/context-only/unsupported`) ③ 오용·misattribution ④ 표절·근접 패러프레이즈 ⑤ attribution
  적정성(특히 SDMAE, 과소·과대). 유령·추정 metadata 금지; 확신 못 하면 "미검증" 상향. **KBS reference 형식**이
  정확히 맞는지도 §A-3 기준으로 확인하되 **DOI·URL은 생략.**
  - **Cadence:** **첫 cycle = 전수조사** — 모든 reference를 하나하나, 최대로 엄격하게(공식 출처·형식 일치 포함).
    **이후 cycle = delta만** — 신규·변경 인용 + 이전 flagged 항목만.
- **P2 CAPTION.** Table·Figure 내용은 **확정 최종본(불변, D4)**이며, 이 프로토콜은
  **각 그림·표의 caption을 정밀 교정**한다: 독립적으로 읽혀도 이해되는 **정보충분·간결·정확한 학술 캡션**
  (§C 적용·명칭 LASAD 일관). 렌더 결과의 overflow·float 배치·페이지 분할 등은 **P3 PDF 시각 QA**에서 점검한다.
- **P3 DUAL-BUILD (둘 다 필수·상호 대체 불가, 둘 다 §A-3로 판정).** (a) **LaTeX 정적 감사** — latexmk 무오류
  컴파일·로그 grep·미해결 citation·label/ref·float·package + FORMAT_SPEC 정적 검증 + 옛 명칭 잔재 grep.
  (b) **PDF 시각 QA** — PDF를 페이지 단위 이미지로 검사: overflow·**float/그림·표 배치**·캡션·페이지 분할·
  참고문헌 스타일·제목·running head·그림 내 명칭·**appendix 가독성**.
  **렌더 규약(§6-SCRATCH 준수·중요):** `pdftoppm`/`latexmk` 등으로 만드는 *전이적* 페이지 PNG·빌드 stdout
  로그는 반드시 신뢰 스크래치 루트 **`/tmp/claude-$(id -u)/lasad/`** 아래에 쓴다 — `paper_writing/.../temp/`나
  임의 `/tmp/<name>`은 금지(매번 권한 프롬프트 유발). 증거로 *보존*할 최종 QA 스크린샷만 `BUILD/`에 둔다.
  양쪽 모두 `Major` 이상 0·형식 위반 0일 때만 통과.
## §5. CONSENSUS GATE (종료 — 가장 중요)
사이클은 **"완성도를 높이는 다양한 sub-agent — 독립 senior-reviewer ≥5 + 전 critic(§A–§G) + adjudication-panel
+ gatekeeper — 가 원고가 top-venue 상위 0.0001%(사용자가 말한 0.001% 이상보다 더 엄격) 기준을 통과했다는 데
명시적·기록된 합의(CONSENSUS)에 도달할 때까지" 무제한 반복**한다. **단 한 명이라도 미해결 `Blocker/Critical/
Major` 이견을 남기면 합의 미성립.** 합의·서명·잔여 이견 0을 `CONVERGENCE_LOG.md`에 기록. **반복 상한 없음.**

합의 성립을 위해 동시 충족:
1. **Loop-until-dry:** 매 라운드 새 독립 senior-reviewer ≥5(이전 결과 비노출), **2연속 라운드** 새 `B/C/M` 0건
   + `derivative-impression-reviewer` 2연속 "SDMAE의 TS판 아님" + attribution 적정.
2. **Rubric ≥9/10(전원), 15차원:** ①Novelty ②Contribution calibration(non-derivative·§G) ③Method soundness
   ④TSAD-domain ⑤Experimental design ⑥Reproducibility ⑦Narrative &
   multi-scale ⑧Register & language(캡션 포함) ⑨수식·표기 ⑩Figure·Table 캡션·배치 렌더(시각 QA) ⑪Reference 무결성(P1)
   ⑫Top-venue 부합 ⑬KBS scope-fit(§A-2) ⑭형식 규격·appendix·float 배치(§A-3) ⑮Naming·title alignment(§F).
3. **Completeness critic**가 누락(미검증 주장·안 읽은 원전·미확정 형식·rename 잔재·SDMAE 과의존·KBS
   name-anchoring/IJUFKS·안 다듬은 캡션·안 본 관점)을 못 찾음.
4. **Coherence/register(D7):** 중복·불일치·난잡화·AI투·비문·명칭 불일치 `Major` 이상 0.
5. **BUILD & VISUAL(P3):** **LaTeX 형식 검증 그리고 PDF 시각 검증 둘 다 통과** — `Major` 이상 0·형식 위반 0
   (appendix·float 배치 포함) **+ 본문 분량 ≤10p 준수(§A-3-LEN)**.
6. **REFERENCE(P1):** 첫 cycle 전수조사 통과 + 이후 delta 해소.
7. **§9 deliverable 3종 산출 완료.**
8. **MASTER-CONTROL 완전 해소(D13·§12):** `MASTER_CONTROL.md`의 **모든 피드백 항목과 모든 불변식**이 최종 pass에서
   green(반영·전파 sweep·전반 통합·이번-pass 재검증 완료, 또는 사유 명시된 의도적 보류)이고, **2연속 dry round**
   동안 회귀(한 번 green이던 항목이 다시 누락/약화) 0. 미반영·미재검증 항목이 하나라도 있으면 CONSENSUS 불성립.
9. **PHASE INTEGRITY & 전수조사(§12-7·§12-8·§12-9):** 모든 loop에서 P1~P6 각 phase가 **생략·축약 없이 실질 수행**
   (`phase-integrity-gate` 통과) + reference **전수조사 실제 완료·검증**(§12-8) + **순차 문단 흐름 감사 통과**(§12-9).
   하나라도 생략·미완료면 CONSENSUS 불성립.

미합의 시 관련 phase 재실행.

## §6. WORKSPACE & ARTIFACTS (`./paper_writing/paper_feedback/` 전용)
```
manuscript/                  # ./paper_writing/paper/ 사본 — 실제 수정 대상 (오직 ./paper_writing/paper/ 에서 복사; 다른 paper* 금지)
00_index/INDEX.md
DECISION_LOG.md
STANDARDS/  TOP_VENUE_STANDARD.md · KBS_FINGERPRINT.md · FORMAT_SPEC.md · NAMING_TITLE_SPEC.md
NEUTRAL_BASELINE.md(271 활성 vs 미사용·배제 2-목록) · NAMING_TITLE_AUDIT.md(§F) · ANCHOR_POSITIONING.md(§G)
FINDINGS_LEDGER.md · REVISION_PLAN.md · REFERENCE_AUDIT.md(P1)
BUILD/  정적감사 로그 + PDF + 시각QA 스크린샷 + 형식/배치 위반 대장(P3)
LANGUAGE_LOG.md · COHERENCE_LOG.md(D7) · CONVERGENCE_LOG.md(§5 점수·dry·CONSENSUS 서명)
result/                      # ← 최종 산출물(§9) 전부 여기에
  ├─ overleaf/  + LASAD_overleaf.zip      # (1) overleaf 업로드용 latex 기반 zip
  ├─ LASAD.pdf                            # (2) 완성된 PDF
  └─ REPORT.md                            # (3) 사용자 보고 요약(채택/기각 근거·CONSENSUS 요약)
```

**§6-SCRATCH (전이적 중간 산출물 — bypass 권한 프롬프트 회피·하드 규약).** `pdftoppm` 페이지 렌더·`latexmk`
stdout 로그 등 *전이적* 파일은 **반드시 `/tmp/claude-$(id -u)/lasad/`**(라운드별로 `/tmp/claude-$(id -u)/lasad/$$/`
권장) 아래에만 쓴다. 이유: `bypassPermissions` 모드에서도 Claude Code는 *git-추적 작업트리 +
`/tmp/claude-<uid>/…` 스크래치* 밖으로 파일을 생성/삭제/`>`-리다이렉트하는 Bash에 확인 프롬프트를 띄운다(모드와
무관한 별도 안전 레이어). `paper_writing/`은 `.gitignore`의 `/paper*/`로 무시되고 `*.png *.pdf *.log`도 전역
무시되므로, 그 아래 `temp/`나 임의 `/tmp/<name>`에 쓰면 매 렌더/빌드마다 프롬프트가 뜬다. **`BUILD/`·`result/`에는
*보존*할 최종물만** 둔다(드물게 1회 쓰므로 무방).
**정적 컴파일도 스크래치에서:** `latexmk`는 `.pdf/.aux/.log`를 `.tex` 옆에 쓰므로 gitignored
`manuscript/.../07_latex/`에서 직접 돌리면(=쓰기) 프롬프트가 뜬다. → 빌드 시 `07_latex/` 전체를
`/tmp/claude-$(id -u)/lasad/build/`로 복사해 **그 사본 안에서** `latexmk` 실행(상대 `\includegraphics`·`.bib` 보존),
거기서 `pdftoppm`으로 페이지 PNG도 렌더하고, 통과한 **최종 `main.pdf`만 `result/`로 1회 복사**한다. (읽기전용
grep/sed/cat은 `manuscript/`에서 그대로 해도 무방 — 쓰기만 스크래치로.) 모든 워크플로/Bash 작성 시 이 규약을 적용하라.

## §7. SUB-AGENT ROSTER (출발점 — D10에 따라 확장·교체·제거 자유)
각 agent에 단일 책임·출처 근거 의무·skeptic 기본값 부여, 가능하면 schema 강제.
- **모델 배정(agent별 · orchestrator 판단).** 각 agent를 띄울 때 작업 성격에 맞는 모델을 지정한다(`agent()`의 `model` 옵션):
  · **`fable`** — 문장 품질·추론·해석·판단이 실질적으로 필요한 agent(예: `senior-reviewer`·`tsad-domain-expert`·
    `methodology-skeptic`·`novelty-positioning-critic`·`experimental-rigor-critic`·`narrative-arc-reviewer`·
    `derivative-impression-reviewer`·`tension-synthesizer`·`adjudication-panel`·`section-rewriter`·`caption-editor`·
    `academic-register-editor`·`anchor-positioning-editor`·`kbs-framing-editor`·`math-notation-auditor`·
    `claim-source-fit-auditor`·`pdf-visual-qa-reviewer` 등).
  · **`opus`** — 단순 확인·기계적·반복 작업 agent(예: `official-format-verifier`·`format-compliance-reviewer`·
    `reference-existence-metadata-verifier`·`kbs-bib-format-auditor`·`latex-static-auditor`·`overleaf-packager`·
    `coverage-drift-gate`·`phase-integrity-gate`·rename/명칭-잔재 grep·형식 정적 검증 등).
  경계가 모호하면 **작업이 실제로 해석·판단을 요구하는지**로 가른다(요구하면 `fable`, 순수 확인·형식이면 `opus`).
- **A. Grounding/기준:** `config-271-archivist`(271 config[experiment_metadata.json]+코드 정독 → 활성/미사용
  2-목록 → NEUTRAL_BASELINE), `top-venue-standards-cartographer`(§A-1),
  `kbs-scope-fingerprint-analyst`(§A-2), `official-format-verifier`(§A-3), `naming-title-alignment-auditor`(§F),
  `anchor-dependence-assessor`(§G).
- **B. 독립 비평(Phase1, blind, §A–§G 채점):** `senior-reviewer ×≥5`, `tsad-domain-expert`,
  `methodology-skeptic`, `novelty-positioning-critic`, `experimental-rigor-critic`,
  `reproducibility-auditor`, `narrative-arc-reviewer`, `caption-critic`(캡션 정보충분·정확·독립가독),
  `math-notation-auditor`, `academic-register-auditor`, `kbs-fit-reviewer`(§A-2), `format-compliance-reviewer`
  (§A-3·appendix·float), `title-method-alignment-reviewer`, `derivative-impression-reviewer`(§G),
  `sequential-flow-auditor`(§12-9 순차 문단 흐름 감사 — 매 loop).
- **C. Reference(P1):** `reference-existence-metadata-verifier`, `claim-source-fit-auditor`,
  `plagiarism-attribution-guardian`, `kbs-bib-format-auditor`(DOI·URL 생략).
- **D. 심의/게이트:** `tension-synthesizer`, `finding-skeptic ×N`, `adjudication-panel`, `gatekeeper`(게이트 + §5 서명),
  `coverage-drift-gate`(§12-4 매 pass END 커버리지·회귀), `phase-integrity-gate`(§12-7 매 loop END phase 무생략 검증).
- **E. 수정/정합:** `section-rewriter ×N`, `citation-integrator`, `caption-editor`(캡션 정밀 교정),
  `layout-appendix-engineer`(컴파일·overflow·페이지분할 렌더 안전 점검 + §A-3 형식 준수 — 그림·표 내용은 불변),
  `coherence-consistency-guardian`(§B+명칭), `academic-register-editor`(§C·§D), `anchor-positioning-editor`(§G),
  `kbs-framing-editor`(§A-2).
- **F. 빌드/패키징(P3):** `latex-static-auditor`, `pdf-visual-qa-reviewer`(appendix·배치 포함),
  `overleaf-packager`(`./paper_writing/paper/` 불변).

## §8. PHASE PIPELINE (게이트 + 자기검토 + CONSENSUS 루프)
> 독립 작업은 fan-out, 교차의존 필요 지점에서만 barrier. **매 phase 끝에 gatekeeper 서명 + D10 self-review.**
> **매 pass(=P1~P6 한 회전) START에 `MASTER_CONTROL.md`+원본 피드백 재확인(직전 pass 결론 비노출), END에
> `coverage-drift-gate` 통과(D13·§12) 필수 — 모든 피드백·불변식의 이번-pass 재검증·누락 0·편향 0을 확인하기 전에는
> 다음 pass나 CONSENSUS로 진입 금지.**
- **P0 Grounding & Standards.** **오직 `./paper_writing/paper/`**→`manuscript/` 사본. A팀 병렬 →
  NEUTRAL_BASELINE(271 config[experiment_metadata.json]+코드 → 활성/미사용 2-목록) +
  §A-1→§A-2→§A-3→§F(spec+1차 sweep)→§G. 참고 Notion 2종은 읽기 전용·검증 대상(D1; 방법론 개요=객관 사실만,
  비교 ground truth=수치 무시). (비평 패널 이전 피드백 비노출.)
  **추가(D13·§12): 원본 피드백 파일을 누락 없이 atomize하여 `MASTER_CONTROL.md`(피드백 ledger + 불변식 체크리스트 +
  ground-truth 사실 핀)를 구축한다. 이미 사전 구축된 `MASTER_CONTROL.md`가 있으면 그것을 단일 권위로 채택하되,
  원본 피드백과 1:1 대조해 누락 0을 검증한 뒤 사용한다.**
- **P1 Independent Review(blind fan-out).** B 패널 동시 채점 → `FINDINGS_LEDGER.md`.
- **P2 Verification & Adjudication.** 통합→skeptic 검증(반박 과반 기각)→adjudication-panel 판정·근거→채택분
  severity순 `REVISION_PLAN.md`.
- **P3 Reference Forensics(P1).** **첫 cycle 전수조사** / 이후 delta → `REFERENCE_AUDIT.md`.
- **P4 Caption(P2).** 각 그림·표 caption 정밀 교정(§C·독립가독) → manuscript. **그림·표 내용은 확정본(불변, D4);** 렌더/배치 점검은 P6 dual-build 시각 QA에서.
- **P5 Revision + Passes.** 채택분 반영 → `caption-editor`+`layout-appendix-engineer`+
  `coherence-consistency-guardian`+`academic-register-editor`+`naming-title-alignment-auditor`+
  `anchor-positioning-editor`+`kbs-framing-editor` 통독 교정·로그.
- **P6 Dual-Build(P3).** LaTeX 정적 + PDF 시각 둘 다 → 결함을 P5로 되돌림. 하나만 통과는 실패.
- **P7 CONSENSUS LOOP(§5).** 새 reviewer ≥5 + derivative-impression + kbs-fit + completeness 재투입 → 결함 시
  P2~P6 재실행 → 명시적 CONSENSUS까지 무제한 반복.
- **P8 Packaging & Delivery(§9).**

## §9. DELIVERABLES — 전부 **`./paper_writing/paper_feedback/result/`** 에 산출되어야 종료
1. **`result/overleaf/` + `result/LASAD_overleaf.zip`** — Overleaf 업로드용 **latex 기반 zip**. §A-3 준수·무오류
   컴파일·LASAD 일관. (`./paper_writing/paper/` 불변.)
2. **`result/LASAD.pdf`** — 완성된 PDF(P3 양쪽 통과: 형식 위반 0·명칭 잔재 0·appendix/float 정상).
3. **`result/REPORT.md`** — 사용자 보고 요약: 채택/부분채택/기각 근거, LASAD 정렬·강조점, SDMAE right-sizing·근거,
   KBS framing, 남은 author-action(title-method 미해결 등), 기준 대비 현 위치, CONSENSUS 요약.

## §10. RED LINES (하드 금지 — 한 줄)
- `./paper_writing/paper/` 수정·추가·삭제 / **`./paper_writing/paper_legacy/`·다른 `paper*` 참고**(D11) · 구체 수치 날조(D3) ·
  형식을 주석·추측으로 검증(D3·§A-3) · 271 미사용 경로를 논문에 언급(method·비교·각주 포함)(D3) ·
  제목·모델명 LASAD 임의 변경(D9) ·
  SDMAE 은폐 또는 distillation 헤드라인화(D8·§G) · KBS fit을 저널 이름으로 논증 / "uncertain
  information processing"(IJUFKS) 어휘(§A-2 ban1) · 유령·추정 reference / 근거 없는 finding 자동 채택(P1·D5) ·
  산출물을 `result/` 밖에 두기(§9) · "이 정도면 충분"식 조기 종료(오직 §5만 종료).

## §11. KICKOFF
(1) **오직 `./paper_writing/paper/`**→`./paper_writing/paper_feedback/manuscript/` 사본 + 작업공간 초기화 → (2) P0 standards 구축
(§A-1→§A-2→§A-3→§F→§G) → (3) P1 blind 비평 → P2~P6(심의·reference 전수조사·caption·
정합·문체·명칭·anchor·KBS·**dual-build**) → **P7 CONSENSUS 루프**. 변경 시 `DECISION_LOG.md`, 매 라운드 각 LOG
갱신. **다양한 sub-agent가 top-venue 0.0001% 통과에 명시적 CONSENSUS에 도달하고 §9 deliverable 3종을
`./paper_writing/paper_feedback/result/`에 산출했을 때만** 종료 보고를 올려라.

## §12. ANTI-DRIFT & MASTER-CONTROL PROTOCOL (D13 운용 — loop 장기화 시 누락·편향 방지)
> 문제: loop가 길어질수록 (i) 피드백 항목·SPEC 주의사항이 조용히 누락되고, (ii) 직전 pass 산출물에 얽매여 성급히
> 판단(이미 됐다고 넘김)하는 경향이 있다. 이를 **외부화된 단일 권위 체크리스트 + 매 pass 양끝 게이트**로 막는다.

- **§12-1 MASTER_CONTROL.md (작업공간 루트, 단일 권위).** 구성:
  - **Part A — 피드백 ledger.** 원본 피드백을 **누락 없이 atomize**한 항목별 행: `ID`(안정 식별자) · 원문(충실 요약) ·
    `CLASS`(아래 §12-2) · 대상 위치(섹션/표/그림) · **근본원인+전파 sweep 범위**(D-INTAKE) · **STATUS**(unaddressed/
    applied/integrated/verified + pass 표시 + evidence file:line). 길이를 이유로 요약·샘플링 금지 — **모든** 항목 등재.
  - **Part B — 불변식 체크리스트.** D1–D15 · §10 RED LINES · §A-2/§A-3/§F/§G 핵심 규칙 · P1–P3 를 각각 **pass별
    이진 게이트 항목**으로 등재(이번 pass 준수=✅, 위반/미확인=❌+사유).
  - **Part C — Ground-truth 사실 핀.** 사용자가 '무조건/확정/ground truth'로 못박은 사실(수식·프로토콜·정정 등)을
    **그대로** 고정 기록해 매 pass 동일하게 반영(재해석·완화 금지).
  - **Part D — 교차참조 그래프.** 같은 근본원인을 공유하는 항목 묶음(전역 일관 수정 단위)을 명시.
- **§12-2 CLASS(항목 처분 유형).** `MANDATORY`(사용자가 무조건/확정으로 지정 — D5 adjudication 면제, 필수 반영) /
  `PROPOSAL`(가설 — D1·D5 심의로 채택/부분/기각) / `UNDERSTAND-FIRST`(의미를 먼저 완전히 규명한 뒤 처분) /
  `PROCESS`(작업 방식 지시 — 절차에 반영). MANDATORY는 판정 금지·충실 반영하되 RED LINE 위반 시에만 surface.
- **§12-3 매 pass START — 재확인(re-ground).** orchestrator와 그 pass의 모든 subagent prompt에 `MASTER_CONTROL.md`
  경로 + 원본 피드백 파일 경로 + §10 RED LINES를 명시해 **각자 직접 Read**하게 한다. **직전 pass 결론·disposition·
  산출물은 이 시점 비노출**(blind). subagent는 자신이 맡은 항목/불변식을 현재 manuscript에서 독립 재유도한다.
- **§12-4 매 pass END — `coverage-drift-gate`.** 전담 agent가 MASTER_CONTROL 전 항목을 훑어:
  (i) 모든 Part A 항목이 **이번 pass에서** 반영·전파·통합·재검증됐는가(evidence 첨부), (ii) 모든 Part B 불변식이
  이번 pass 준수됐는가, (iii) 회귀(전 pass green→이번 pass 누락/약화) 0인가 를 판정. **하나라도 ❌면 pass 미종료
  → 책임 owner 재투입.** 결과를 `MASTER_CONTROL.md` STATUS와 `CONVERGENCE_LOG.md`에 기록.
- **§12-5 ANTI-ANCHORING(D1·문제 2a 직접 대응).** "직전 loop 산출물에 의존한 성급 판단" 금지: 매 pass는 모든 항목을
  **다시 연다**. '해결됨' 표시는 *이번 pass 재확인의 대상*일 뿐 면제 사유가 아니다. 비평 패널은 이전 disposition을
  보지 않고 현재 텍스트만 근거로 판단한다.
- **§12-6 CONSENSUS 연동.** §5-8(MASTER-CONTROL 완전 해소 + 2연속 dry 회귀 0)을 충족하지 못하면 종료 불가.
- **§12-7 LOOP 정의 & NO-SKIP PHASE INTEGRITY (loop = 피드백 phase 전체의 엄격 반복; 생략·대충 넘김 금지).**
  1 loop = **P1→P6 한 회전을 빠짐없이, 직전 loop 산출물에 anchor 없이(§12-5) 처음부터 끝까지 실질 수행**하는 것.
  어떤 phase도 생략·축약·rubber-stamp("이전과 동일"·빈 산출물·"생략")로 넘기지 마라. 각 phase는 그 loop의
  **날짜 표시 실질 artifact**를 남긴다. 매 loop END에 `phase-integrity-gate`가 P1(blind 비평)·P2(adjudication)·
  P3(reference)·P4(caption)·P5(revision)·P6(dual-build) **각각이 이번 loop에서 실제로 실질 수행됐는지** 확인 —
  미수행·축약 phase 발견 시 그 phase 재실행 후에만 loop 종료(coverage-drift-gate와 별개의 hard gate).
- **§12-8 REFERENCE 전수조사 강제(§5-6 보강).** 첫 cycle reference 전수조사는 반드시 **실제 완료·기록**되어야 한다 —
  각 reference에 대해 (i) 본문 실사용 여부 (ii) metadata 완벽 일치 (iii) 인용 맥락이 실제 reference 맥락과 부합
  (iv) Elsevier/KBS reference format 준수 (v) 오용 없음 을 `REFERENCE_AUDIT.md`에 **ref별로** 표기(URL·DOI 생략,
  §A-3 형식 기준). 전수조사 미완료 상태로는 어떤 loop도 'reference 완료'로 표기 불가·CONSENSUS 불가. **아주 작은
  오류도 불용.**
- **§12-9 순차 문단 흐름 감사(`sequential-flow-auditor`).** 매 loop에 원고를 **처음부터 문단 단위로 순차** 통독한다
  (전체를 한꺼번에 보지 말고 글 순서대로 한 문단씩). 점검: 미설명 개념의 갑작스런 등장·앞 내용과의 비일관·
  용어/표현 비통일·흐름 단절. 발견은 `FINDINGS_LEDGER`+`MASTER_CONTROL`에 반영해 그 loop 안에서 처리.
