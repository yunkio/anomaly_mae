---
phase: 6
agent: style-rechecker
round: r2
directives: [T6, R4]
last_modified: 2026-06-11
target: paper/05_manuscript/MANUSCRIPT_v3.md (diff base: MANUSCRIPT_v2.md)
fixlog_under_review: paper/99_reviews/p6_style_fixlog_r1.md
method: |
  ① python difflib line-diff + sentence-level diff (101 hunks, 229 changed v3 sentences, 전수 검토)
  ② MUST급 마감을 fixlog 주장이 아닌 v3 본문 grep/실측으로 확인
  ③ 변경 문장 전수를 SENTENCE_CORPUS.md(부록 A/B 포함) 기준으로 재검
  ④ 마커/인용/수식 multiset 기계 대조 (frontmatter 제외 본문 기준)
  ⑤ 거부 처리 사유를 정본(trainer.py/model.py/EXPERIMENT_PROTOCOL_TRUTH/unified_loader.py/
     PLACEHOLDER_REGISTRY FIG-2·TAB-1 spec)과 직접 대조 (코드 read-only 열람만)
verdict: "PASS — BLOCKER 0 / MAJOR 3 / MINOR 5. MUST급 전건 실해소·마커 무손상·신규 의미 훼손 없음. MAJOR 3건은 국소 문장 수술로 해소 가능 (게이트 차단 아님; r2 touch-up 권고)."
---

# P6 수정분 재검사 r2 — style-fixer r1 검증

## 1. MUST급 마감 표 (본문 실측 기준)

### 1.1 AI_PHRASING_LEDGER — MUST 11/11 해소 확인

Priority Fix List(§V) 기준 11건. (참고: C-01은 본문 entry에 SHOULD로 표기되어 있으나 §V에 MUST로 등재 — fixer가 MUST로 처리한 것은 보수적·타당.)

| ID | v3 실측 근거 | 판정 |
|---|---|---|
| I-01 | L158: 첫 문장 em-dash splice 제거, "multi-channel" 이중수식 삭제 — **ledger의 Revised sentence를 그대로 채택** | 해소 |
| I-03 | L163: "share an implicit assumption" → "treat the training data as drawn entirely from normal operations" | 해소 |
| I-04 | L164–165: colon-launch 제거; "architectural pathway for leveraging…" → "no mechanism for exploiting…"; 문장 분리 | 해소 |
| I-06 | L169–171: 69-word 융합문 → 3문장; "data protocol" → "data partition"; stranded preposition 제거 | 해소 |
| I-08 | L174: 결과 단정 → "is designed to amplify … (Section 4.3)" 설계 의도 + 포인터 (A8 정합) — 단, 신규 어색함 R2-M1 참조 | 해소(어색함 별건) |
| A-01 | L114 Abstract: "loss bifurcation between … paths" → "a Student imitation loss restricted to normal patches"; 3항 병렬 복구 | 해소 |
| RW-06 | L213–214: 60-word PU 문장 분리 ("Established solution families include …") | 해소 |
| RW-08 | L215: → "methods that incorporate anomaly labels into the representation learning objective itself are rare" | 해소 |
| M-03 | L257–258: "underpinning" 제거; "receive a stop-gradient copy"; 문장 분리 | 해소 |
| E-05 | L455–456: §4.2 4절 복합 결과문 → 2문장 (placeholder 5개 모두 보존) | 해소 |
| C-01 | L532–533: "underexplored setting" → "the contaminated semi-supervised setting" 명명 + 분리 + 현재시제; "unsupported" → "unaddressed" | 해소 |

### 1.2 STYLE_AUDIT_A — 명시적 MUST-FIX 17/17 해소 확인

(summary 표는 25를 주장하나 `Severity: MUST-FIX` 표기는 17건 — grep 실측. fixlog의 카운트 노트와 일치. 17건 전건 처리 확인.)

| ID | v3 실측 근거 | 판정 |
|---|---|---|
| A-004 | L115: 소유격 명사화 제거 — "the capacity-limited Student mimics the Teacher less faithfully on anomalous … than on normal ones" (B-012 비교급 모호성 동시 해소) | 해소 |
| S1-002 | L159–160: "and because" 분리; "has been" 유지 (§5-C3 — 타당) | 해소 |
| S1-005 | L169–171 (I-06과 동일 지점) | 해소 |
| S1-009 | L190: "fails more severely on anomalous correlation patterns than on normal ones" | 해소 |
| S2-004 | L213–214 (RW-06 동일) | 해소 |
| S3-006 | L280: "position embeddings **are** added, and the full sequence **is** passed" | 해소 |
| S3-008 | L294–296: 101-word dual-λ 문장 → 3문장; B-036 EMA 미수입 (§3.1 검증 참조) | 해소 |
| S3-010 | L320–321: 90-word → 2문장; "takes as input … and predicts"; "focal-style binary cross-entropy (BCE) variant" | 해소 |
| S4-001 | L374: 6개 family 전부 명명 (SWaT, WaDi, PSM, SMD, SMAP, MSL); "entities"/"evaluation conditions"; **audit의 오판(WaDi A1/A2=2 family) 교정 — 정본 일치 확인** (§4.1 참조) | 해소(교정 타당) |
| S4-010 | L419–421: 88-word 임계값 문장 → 3문장; "follows the … mechanism introduced by" 인용 강도 보존; R30 서사 무결 | 해소 |
| S4-011 | L425–429: 155-word 지표 문장 → 지표당 1문장 (5+1); "complementary under class imbalance" 보존; liu2024elephant 주장 범위 보존 ("time-series anomaly detection"으로 spell-out) | 해소 |
| S4-014 | L445: "the surviving normal segments **are** concatenated" | 해소 |
| S4-015 | L451: "per-entity results **are** in Appendix §A.6" | 해소 |
| S5-001 | L532: "This paper **addresses**" | 해소 |
| S5-002 | L534: "We proposed" 제거 → "CSMAD integrates …"; "on top of" → "built on" | 해소 |
| SA-004 | L606: "$K = 100$ **recovers** point-wise scoring" | 해소 |
| SC-001 | L749: "$[e_0, e_1]$ **is** the student-training phase … $\tau$ **is** its normalized progress" | 해소 |

### 1.3 STYLE_AUDIT_B — Moderate 3/3 해소 확인

| ID | v3 실측 근거 | 판정 |
|---|---|---|
| B-012 | L190 (S1-009와 동일 해소); "a design intended to make" 유지 — A8(결과 미충전) 정합, audit의 단정형 거부 타당 | 해소 |
| B-046 | "learning units" 본문 0건 (grep); L374 "113 entities … 114 evaluation conditions"; L552 "113 entities"; registry sync-group A 갱신 확인 | 해소 |
| B-063 | L534: "loss bifurcation **that restricts Student mimicry to normal patches**" — "bifurcation toward" 비문 제거, 정의된 용어 유지 | 해소 |

### 1.4 TERMINOLOGY_AUDIT 핵심 — 전건 해소 확인

| 항목 | v3 실측 | 판정 |
|---|---|---|
| Q1/Q3 11곳 (HIGH) | 본문 `\bQ[13]\b` **0건** (잔존 4건은 모두 L94 이전 YAML frontmatter history). §4.1.4에서 **anomaly-excised condition** / **contaminated-training condition** 양 용어 bold 최초 정의 (L445, L447); weakly-supervised 문장은 "(defined below)" 전방참조 (L441); §B.1 제목/표/Δ 캡션 개명 (L699/704/706); registry v3-r1 동기화 (TAB-2 L102, TAB-B1 L205–206, 잔존 Q1/Q3는 code-label 병기·audit-trail 명시뿐) | 해소 |
| **의미 비반전 검증** | EXPERIMENT_PROTOCOL_TRUTH.md L116–117 (Q1=full=anomaly 포함 / Q3=normalonly=region 제거) + `comparison/data/unified_loader.py` docstring "'normalonly': Remove anomaly regions from training data" + 동 truth L123 "weakly-supervised는 Q1 전용(normalonly에선 train 라벨 전부 0)" ↔ v3 L441 논리 일치. **fixlog §0.2의 briefing-가설 기각(C12)도 타당** — briefing 예시 매핑이 §4.2 protocol-effect 축과 혼동된 역방향이었음 | 비반전 확정 |
| excl22 (MED) | 본문 최초 사용 L395에서 정의 ("a condition denoted excl22"); 이후 전 용례(L396, 457, 642, 674, 682–685…) 정의 후행; §A.4 제목 spell-out (L674); §4.2 산문 "under SWaT's excl22 condition" (L457) | 해소 |
| TSAD (MED) | bare TSAD **0건**; MTSAD 통일 + 일반-분야 주장 2곳 spell-out (L361 liu2024elephant, §2.2 NRdetector) — 인용 주장 범위 보존 | 해소 |
| d_model (MED) | `d_\text{model}` **0건**; `d_{\mathrm{model}}` 3개소 (L401, L559 Table A.1, L780 Table C.1 주변) | 해소 |
| gradient-reversal 하이픈 | 전명사형 "gradient reversal suppression" 0건; 하이픈형 7건; bare 명사구 "gradient reversal" 유지 | 해소 |
| BCE 최초 산문 정의 | L321 "focal-style binary cross-entropy (BCE) variant" — 이전 산문 사용 없음 | 해소 |
| Table C.2 +4 기호 | $\bar{r},\bar{d}$ / $\varepsilon$ / $c$ 행 추가 (L824–826); **값 검증**: Eq. 4 ($\varepsilon = 10^{-4}$) · Eq. 5 ($c = 4$) · Table A.1 "Combination ratio $c=4$" 모두 일치 | 해소 |
| Teacher/Student 대문자 규칙 | L226 각주에 선언문 삽입 — 위치·내용 적절 | 해소 |

## 2. 신규 발견 (변경 문장 전수 corpus 재검)

### MAJOR (신규 어색함 — r2 국소 수정 권고)

- **R2-M1** | §1 L174: "**Exploiting all three simultaneously is designed to amplify** both the reconstruction error and the Teacher–Student discrepancy at anomalous regions (Section 4.3)." — 동명사 주어("Exploiting")에 "is designed to"를 결합한 행위주성 불일치. corpus의 "is designed to"는 설계물 주어 한정 (DCdet "A … structure is designed to learn"). I-08의 결과단정 제거 취지는 옳으나 fixer 자작 구문이 새 어색함을 만듦. 제안: "The design exploits all three simultaneously to amplify …" 또는 "Exploiting all three simultaneously is intended to amplify …".
- **R2-M2** | §A.1 L590: "… comprise nine detectors adopted from the protocol study of \cite{sarfraz2024quovadis} — five simple detectors (…), three lightweight neural detectors (…), and a GCN-LSTM detector **— six established deep MTSAD systems (…), and seven recent methods (…)**." — B-065(dangling "following") 수정이 만든 신규 파싱 곤란: dash-pair 닫힘이 외부 열거(nine/six/seven)의 구분자를 겸하면서 "a GCN-LSTM detector — six established …"가 동격 연결로 오독됨. v2는 외부 열거를 semicolon으로 구분했음. 제안: 내부 분해를 괄호로 내리고 외부를 semicolon 복원 — "nine detectors adopted from the protocol study of \cite{…} (five simple detectors: …; three lightweight neural detectors: …; and a GCN-LSTM detector); six established deep MTSAD systems (…); and seven recent methods (…)".
- **R2-M3** | Abstract L117: "we **introduce** a contaminated benchmark protocol that incorporates …, thereby **introducing** labeled anomalies …" — A-005 적용 시 audit 제안어 "exposing"을 "introducing"으로 바꿔 한 문장 내 introduce/introducing 반복(echo)을 신설. Abstract 가시성이 높아 MAJOR. 제안: audit 원안 "thereby exposing" 복원 (또는 "thereby adding").

### MINOR (다듬기 — Phase 7 일괄 가능)

- **R2-m1** | §4.1.3 L434: "**labelled** (oracle)" — 영국식 철자. 원고 표준은 "labeled" (53건). v2의 유일 선례(L532 "labelled \"DAGMM (simplified)\"", 기존분)를 따라 신규 1건 추가됨 — 둘 다 "labeled"로 통일 권고.
- **R2-m2** | §2.3 L220: "… constitute independent developments, **and** our design follows directly from vision MAE." — v2의 dash가 담던 대조("independent developments — **ours** traces to vision MAE")가 "and" 등위로 약화. "whereas our design follows …" 권고.
- **R2-m3** | §4.1.1 L387: "with the largest shift **being** 166 timesteps" — "with X being Y" 절대구문은 다소 구어적. "(largest shift: 166 timesteps)" 권고.
- **R2-m4** | §A.2 L620: "Affiliation precision and recall **measure** the temporal proximity …, **converted into** per-event affinity scores …" — 분사 "converted"의 부착이 느슨 (v2의 능동 "convert … into …"가 더 단단했음). 재고 권고.
- **R2-m5** | §2.2 L215: "neither model employs a … pretext **or** adversarial …" — 부정 극성 하의 "or"는 문법적이나 "nor"가 격식상 우월. 선택적.

### 검토 후 비문제 판정 (기록)

- §1 첫 문장 "sensor streams **from** water treatment plants …, all of which depend on …" — ledger I-01의 Revised sentence 원문 그대로이며 "all of which"의 선행사(설비들)가 의미상 정확. 수용.
- "\cite{he2022mae} demonstrates" vs "Ristea et al. \cite{ristea2024sdmae} adapt" — 주어가 각각 "The masked autoencoder (MAE) of He et al."(단수)와 "Ristea et al."(복수)로 수일치 정확. 비문제.
- §3.1 L246 "Recovering the **normal** multi-channel correlation structure" — v2 대비 "normal" 1어 추가. §3 전반의 "normal correlation structure" 용법과 일치하는 명료화로 의미 훼손 아님 (기록만).
- §5 L537 future-work 구체화("amortized inference with learned masking schedules or sparse patch selection") — ledger C-02의 Revised sentence에서 유래(fixer 창작 아님); 추측성 hedge 유지. 수용.
- "This label gap" (L169) — 콜론 후행 정의가 자기완결적 (v2 "The gap"보다 개선). 수용.
- em-dash 정책: 본문 em-dash 문자 105→54; 잔존은 열거-동격 pair(§4.1 dataset 열거, §1 "— a source of contamination —" 등)와 placeholder spec 주석. 절-접속 splice 부활 없음. **보상성 남용 부재 실측**: thereby 0→2, Consequently 0→2(+소문자 1), however 2→4, semicolon 153→166, Moreover/Furthermore/Additionally 0→0. corpus B.2 한도 내.

## 3. 무손상 검증 (기계 대조: v2 ↔ v3, frontmatter 제외 본문)

| 항목 | v2 | v3 | 판정 |
|---|---|---|---|
| PH:NUM | 31 (ID 001–031, 전 ID 일치) | 31 (동일 multiset) | **무손상** |
| PH:TXT | 4 occurrences (TXT-001/002) | 4 (동일) | **무손상** |
| PH:FIG | 5 (FIG-1,2,3,4,B1) | 5 (동일) | **무손상** |
| PH:TAB | 11 (1,2,3,A3,A6,A7,A8,B1–B4) | 11 (동일) | **무손상** |
| PH:ALG | 1 (ALG-C1) | 1 (동일) | **무손상** |
| `\cite`/`\citet` 명령 | 89 | 89 | **무손상** |
| cite key 연인원 | 131 | 131 | **무손상** |
| `[X.XX]` / `[N]` 토큰 | 20 / 13 | 20 / 13 | **무손상** |
| 수식 `\tag` | 1–6, C.1–C.5 (11) | 동일 순서·동일 번호 | **무손상** |
| `$$` 블록 | 11 | 11 | **무손상** |
| Highlights 길이 | — | 109/120/120/121/125 (≤125 전건) | 충족 |

(fixlog §7의 "FIG 5 / ALG 1" 주장은 정확 — FIG-B1·ALG-C1 포함 계수. 본 재검의 초기 정규식 누락분 보정 후 일치 확인.)

## 4. 거부 처리 사유 spot 검증 (정본 대조)

| 건 | fixlog 사유 | 정본 실측 | 판정 |
|---|---|---|---|
| **B-036** (EMA) | "previous epoch's plain average — EMA 아님" | `trainer.py:912` (per-batch 누산) → `:937` `epoch_losses[key] /= len(self.train_loader)` (epoch 단순 평균) → `:1319` `_prev_epoch_grl_lambda` 저장 → `:762` 차기 epoch 적용. 지수가중 없음. Eq. C.4 (clip 0–10, $+10^{-4}$, $\beta_{GRL}=0.2$) ↔ `271_CONFIG_TRUTH` L123 `grl_loss_weight=0.2`·L298 일치. (config의 `use_teacher_output_ema`는 별개 default-off 기능 — 혼동 없음) | **거부 타당** (EMA 수입 시 사실 오류였음) |
| **B-031/B-039** (Student encoder) | "GRL은 Student **decoder** hidden states에 부착" | `model.py:1140` `student_hidden = self.student_decoder(…)` → `:1153` `cls_logits = self.anomaly_classifier(student_hidden, lambda_grl)`; registry FIG-2 spec L68 "the Student decoder's final-layer hidden states, before the output projection". v3 L255 합성문 "couples the Student decoder's hidden states …" 정본 일치 | **거부·교정 타당** (audit 원안 수용 시 사실 오류) |
| **C11/S4-001** (family 산법) | "WaDi 1 family(A1/A2 entities); 113=1+2+1+28+54+27" | registry TAB-1 spec L90–96: 6 family rows (SWaT, WaDi A1/A2, PSM, SMD ×28, SMAP ×54, MSL ×27); Table A.4 L647 "SMAP (×54)"; L663 "all 54 channels"; 81 channels = 54+27 (L387) 정합 | **교정 타당** (audit 원안이 Table 1과 모순) |
| **C12** (Q1/Q3 briefing 기각) | truth doc 우선 | §1.4 참조 — 비반전 확정 | **타당** |
| B-006 ("has been"→"is") | 역사-현재 지속 주장 | "the dominant paradigm … has been unsupervised learning" — 현재완료가 의미 정확 | 타당 |
| B-054 ("absolute points") | 지표가 [X.XX] 소수 보고 | NUM-007/008 스펙 소수형 — "percentage points"는 오기였을 것 | 타당 |
| I-10/E-03/B-012/B-059 claim-strength 거부 | A8 (결과 미충전) | v3에 "competitive"(L511)·"a design intended to"(L190)·"prevents … going undetected"(L433)·"unsupervised floor"(L512) 보존 확인 | 타당 |

## 5. 판정

- **BLOCKER: 0** — MUST급(AI 11 + STYLE_A 17 + STYLE_B Mod 3 + TERM 핵심 전건) 실해소 확인; placeholder/cite/수식 무손상; Q1/Q3 의미 비반전 정본 확정; 신규 의미 훼손 없음.
- **MAJOR: 3** — R2-M1 (L174 "Exploiting … is designed to"), R2-M2 (L590 baseline 열거 구두점 붕괴), R2-M3 (Abstract introduce/introducing echo). 셋 다 국소 문장 수술로 해소 가능하며 의미·마커 무관.
- **MINOR: 5** — R2-m1–m5 (labelled 철자, 대조 약화, 절대구문, 분사 부착, nor). Phase 7 polish로 이월 가능.

**게이트 판정: PASS (조건부 권고 동반).** style-fixer r1의 작업 품질은 높음 — 특히 audit 원안의 사실 오류 3건(B-036 EMA, B-031/039 encoder, S4-001 family 산법)을 정본 대조로 걸러낸 것이 검증됨. MAJOR 3건은 style-fixer의 단건 r2 touch-up (예상 변경 3문장, 마커 무관)을 권고하되 Phase 6 진행을 차단하지 않는다.
