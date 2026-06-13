---
phase: 5
agent: integrator
version: v1
directives: [T5, R6]
last_modified: 2026-06-11
inputs:
  - 05_manuscript/sections/{front_intro_conclusion, related_work, method, experiments}.md
  - 03_blueprint/PAGE_BUDGET.md (분량 단일 정본)
  - 04_references/refs.bib (유효 key 49 정본)
outputs:
  - 05_manuscript/MANUSCRIPT_v1.md
  - 05_manuscript/PLACEHOLDER_REGISTRY.md
  - 05_manuscript/INTEGRATION_REPORT_v1.md (본 문서)
  - 섹션 파일 정정 2건 (experiments.md 인용 key, method.md figure 번호 — 주석 명기)
---

# INTEGRATION REPORT — MANUSCRIPT v1 (Phase 5)

## 1. 작업 범위

섹션 드래프트 4편을 Title/Abstract/Keywords/Highlights → §1–§5 순서로 `MANUSCRIPT_v1.md`로 통합.
수행 항목: ① 인용 key 전수 검증, ② placeholder 전역 재부여 + 표기 통일, ③ 용어 통일·중복 제거·transition 보정, ④ R6 분량 체크 + 3회 압축 패스 (directive 의무 서술은 압축만, 삭제 0건), ⑤ PLACEHOLDER_REGISTRY 구축.
수치 창작 0건 (A8): 본문의 모든 실수치는 섹션 드래프트/PROTOCOL_TRUTH 유래; 유일한 placeholder 해소(50×)는 프로토콜 상수 N=50에서 유도 (§4-13 참조).

## 2. 인용 key 검증 (refs.bib 49 keys 대조)

- 스캔: 4개 섹션 파일 `\cite{}`/`\citet{}` 106회(중복 포함), 통합 원고 94회 / 고유 44 keys.
- **무효 key 적발·정정: 1건** — `han2023catch` → `wu2025catch` (CATCH, Wu et al., ICLR 2025; 기지 이슈와 일치). experiments.md 본문 인용 1곳 + Citation Keys Used 목록 1곳, 총 2개소 정정 (섹션 파일에 주석 명기 후 원고에 반영).
- `[CITE-UNRESOLVED]` 마킹: **0건** (정정 불능 key 없음).
- 위양성 1건 기각: related_work.md의 `\cite{key}`는 drafter HTML 주석 내 표기 예시 — 실인용 아님.
- 미사용 유효 key 5건 (Phase 6 인용 기회로 기록): `xiong2020prenorm` (§3.4 Pre-LN 서술의 자연 근거 후보), `xu2023rosas`·`wang2022hscl` (§2.2 contamination-resilient SSL 보강 후보), `darban2024dacad`, `jacob2021exathlon` (Exathlon 제외 방침 R33과 정합 — 인용 불필요 가능성 높음).

## 3. Placeholder 전역 재부여 (충돌 해소)

front(NUM-001~007)와 experiments(NUM-001~023)의 ID 충돌을 문서 순서 기준 전역 유일 ID로 재부여. FIG-1~4 / TAB-1~4는 블루프린트 번호 유지.

| 구 ID (파일) | 신 ID | | 구 ID (파일) | 신 ID |
|---|---|---|---|---|
| front NUM-001/002 (Abstract) | NUM-001/002 | | exp NUM-009~014 (§4.2 Table 4) | NUM-014~019 |
| (신설) Highlights "six"→[N] | NUM-003 | | exp NUM-015~020 (§4.3) | NUM-020~025 |
| front NUM-003/004 (§1 bullet 4) | NUM-004/005 | | exp NUM-021~023 (§4.4–4.5) | NUM-026~028 |
| exp NUM-001~008 (§4.2) | NUM-006~013 | | front NUM-005/006 (§5) | NUM-029/030 |

- front NUM-007 (§5 추론 비용 배수)은 해소 (§4-13). method.md [NUM-r_m]/[NUM-arch]/[NUM-c]는 §4.1.2/§3.6에 실수치로 실현된 config 상수 — registry §5에 audit trail로 기록.
- 표기 통일: 수치 `[X.XX]`/`[N]` + 인접 `<!-- PH:NUM-### | 설명 -->` (원 draft 일부는 가시 토큰 없이 주석만 존재 — 전부 가시 토큰 부여). 신규 TXT-001(GPU)/TXT-002(코드 URL ×3) 등록 — 원 draft에서 미등록이던 인라인 placeholder 2종을 registry에 편입 (누락 0).

## 4. 통합 중 발견한 모순·중복과 처리 내역

1. **[모순/번호 충돌] 아키텍처 figure 번호**: method.md가 [FIG-1]/"Figure 1"로 표기 — 블루프린트 정본은 Fig.1=§1 설정비교, Fig.2=§3.2 아키텍처. → 섹션 파일+원고 모두 FIG-2로 정정. §1에는 [FIG-1] 마커가 본문에 없었음 → Para 3 직후 삽입 (PAGE_BUDGET 배치 지침).
2. **[모순] §3.2 "four functional blocks" vs Fig.2 스펙 "five color regions"**: Teacher/Student를 1블록으로 묶은 본문 표현이 figure 5분할과 불일치 → "five functional blocks"로 정정 (섹션 파일 동반 정정).
3. **[모순] §3.3 마스킹 시점 서술**: "masked before the encoder" 직후 "Masking is performed *after* the encoder stage"로 자가당착 → "withheld from the encoder … mask tokens inserted just before each decoder"로 재서술 (의미 보존: 선택은 인코딩 전, 토큰 삽입은 디코더 직전).
4. **[중복] §1↔§4.1.1 벤치마크 무라벨 enumeration**: 데이터셋별 열거가 양쪽에 중복 → §1은 1절 요약+§4.1.1 포인터로 압축, §4.1.1을 정본 유지 (R13 방어 구조 보존).
5. **[중복] §2.3 각주 ↔ §3.4 각주**: SDMAE 차이(용어 계보·branch-off·GRL 부재)가 두 각주에 중복 → §3.4 각주 삭제, §2.3 각주를 정본(R21 방어 정위치) 유지; §3.5의 "target/loss space vs gradient space" 1문장(의무)은 본문 유지.
6. **[중복] leave-one-out 비용 4중 서술** (§3.6/§4.1.2/§4.2/§5): 메커니즘은 §3.6 정본, §4.1.2는 1문장 포인터(+Appendix §B.3), §4.2의 비용 문단 삭제, §5는 한계 서술로 유지.
7. **[허위 전방참조] §4.2 "§4.3 examines whether complementary masking (7 patterns) can reduce this cost"**: §4.3에 해당 분석 없음 (complementary masking은 §5 "implemented but not used" 향후과제가 정본) → 문장 삭제.
8. **[문장 결함] §4.2 "On SMD, SMD per-machine results…"** → "per-machine SMD results are in Appendix §A.4"로 정정.
9. **[오참조] §4.1.1 "114 evaluation units (§4.1.3)"**: dual-eval 설명은 §4.1.1 내부에 있음 → "(below)"로 정정.
10. **[내부 식별자 노출 — front 정책 위반] 3건**: `affiliation_f1_ar` → "computed at the anomaly-ratio threshold"; `use_grl=True` → "configuration held fixed (label-dependent pathways self-deactivate)"; "force-mask anomaly"/"force-masking" → **"anomaly-priority masking"** (abstract/§1/§3 용어로 §4.2/§4.3/§4.4 + Table 3 row 3 라벨 통일).
11. **[용어 통일] Teacher–Student**: experiments의 소문자 "teacher-student"(hyphen) → 컴포넌트 고유명 "Teacher"/"Student" + 복합어 en-dash "Teacher–Student"로 §4 전체 통일. "six evaluation sets" → "six dataset families" (§4.2; Table 2 열은 WaDi 분리로 7개 — families 기준 서술로 고정).
12. **[중복] §4.1.1 prose ↔ Table 1 차원 수**: 데이터셋별 feature 수가 본문·표에 이중 기재 → prose에서 제거, Table 1 #Dimensions 열을 단일 출처화 (SWaT 45-feature 재현성 주의는 §4.1.2 유지 — ADV MAJ-003 의무).
13. **[비일관 해소] §5 추론 비용 placeholder vs §4.2 실수치**: 구 draft가 §4.2에선 "approximately 50×"를 실수치로, §5에선 placeholder로 적음 → §5를 "approximately 50×"로 통일 (프로토콜 상수 N=50 유도값; 수치 창작 아님). wall-clock 검증은 Appendix §B.3 잔존 — 실측 괴리 시 §5 표현 완화 (registry §5에 조건 명기).
14. **[비일관 해소] Highlights "six" 하드코딩 vs Abstract placeholder**: front 정책(완료 확정 전 placeholder)에 맞춰 Highlight 5의 "six" → [N] (NUM-003). §4 본문의 "six families / 113 units / 26 baselines"는 drafter-4의 명시 정책(프로토콜 상수 실수치)대로 유지 — **front placeholder ↔ §4 상수의 동기화 의무를 registry sync group A/B로 명문화**.
15. **[중복 완화] NRdetector 서술 §1/§2.2/§4.1.1/§4.2 4개소**: §2.2를 정본(포지셔닝 문단)으로 유지, §1은 동기 1문장, §4.1.1은 프로토콜 선례 1문장, §4.2는 비교 1절로 각각 비중 차등화 — 서술 중복(파이프라인 구조 설명 반복)은 §4.2에서 "(Section 2.2)" 포인터로 대체.
16. **[기타]** §4.4 "region-level relabeling" 오해 소지 표현 → "a uniformly random selection of regions retains labels"로 재서술. §3.1 "three-path integration" 선행 미정의 → "the masking, loss, and gradient pathways of Sections 3.3–3.5"로 명시.

## 5. R6 분량 체크 (PAGE_BUDGET §0.1/§8 환산 기준: 본문 ~675 words/p, display 수식 0.05p, figure/table은 §3 사양)

### 압축 경과 (prose 단어수; 주석·placeholder 블록 제외)

| 섹션 | 원 draft | 통합 1차 | 최종 (3차 압축 후) | §8 권장 단어수 | 잔여 초과 |
|------|--------:|--------:|------------------:|---------------:|----------:|
| §1 Introduction | 903 | 743 | **743** | 650–750 | ✓ 범위 내 |
| §2 Related Work | 965 | 737 | **737** | 700–780 | ✓ 범위 내 |
| §3 Methodology | 2,406 | 1,537 | **1,444** | 1,000–1,100 | +344 (~1.3×) |
| §4 Experiments | 3,950 | 2,437 | **2,266** | 780–870 | +1,396 (~2.7×) |
| §5 Conclusion | 262 | 223 | **223** | 150–180 | +43 (≈0.06p, 허용) |
| **합계** | **8,486** | | **5,413** | 3,280–3,680 | −36% 압축 달성 |

### 섹션별 페이지 추정 (최종)

| 섹션 | 텍스트(p) | 수식(p) | Figure/Table(p) | 헤더(p) | **추정 합** | 예산 | Δ |
|------|---------:|--------:|----------------:|--------:|-----------:|-----:|---:|
| §1 | 1.10 | — | 0.40 (Fig.1) | 0.05 | **1.55** | 1.6 | −0.05 ✓ |
| §2 | 1.09 | — | — | 0.09 | **1.18** | 1.1 | +0.08 ✓ |
| §3 | 2.14 | 0.60 (12개) | 0.40 (Fig.2 5cm 가정) | 0.18 | **3.32** | 2.7 | **+0.62** |
| §4 | 3.36 | — | 1.86 (T1 .28+T2 .50 landscape+T4 .20+T3 .25+F3 .33+F4 .30) | 0.20 | **5.42** | 3.3 | **+2.12** |
| §5 | 0.33 | — | — | 0.03 | **0.36** | 0.3 | +0.06 ✓ |
| **본문 합계** | | | | | **≈11.8p** | **9.0p** | **+2.8p (+31%)** |

(Front matter 별도: Abstract ~209w ≈ 0.30p + Keywords 0.03p + Highlights 0.10p ≈ 0.43p — 예산 0.39p와 정합.)

### 판정과 정직 보고

- **§1·§2·§5: 예산 통과** (±0.1p 이내).
- **§3·§4: 압축 후에도 예산 초과** — 본문 합계 추정 ≈11.8p로 목표 9.0p ±10%(8.1–9.9p) **밖**. 3회 압축 패스로 원 draft 대비 −36%(§4 −43%, §3 −40%)를 달성했으나, **directive 의무 서술의 구조적 하한**이 §8 단어 예산과 양립하지 않음을 확인:
  - §4 의무 요소 최소 구성(R13 5-논거 프로토콜 방어 + R28 SWaT dual + epoch 비대칭·test-set selection 공개 + R29 5지표 상보성 + R31 Q3 공정성·양적 비대칭 + R30 protocol-effect 2-논거 + §4.3 컴포넌트 설명 + R32 3-성질 강건성 논리 + NRdetector 축 구분)만으로 ~1,800–2,200 words가 필요 — §8의 780–870 words는 이 목록과 동시에 만족 불가능. PAGE_BUDGET §2 자체의 §4 소절 합산(~1,120w)과 §8(780–870w)도 상호 긴장 관계.
  - §3은 수식 12개 + R23 GRL 필요성 논증 + dual-λ 구조 + warmup 공개를 유지하는 한 ~1,400w가 실질 하한.
- **삭제 금지 원칙 준수**: R13/R28/R29/R30/R31/R32 의무 서술, GRL 필요성 논증, 공개(disclosure) 항목은 전부 잔존 (압축만).

### Phase 7 조정 여지 (기록 — 오케스트레이터 재가 필요 항목 표기)

1. **[재가 불요] float 추가 절약**: Table 4의 Table 2 흡수(−0.15p, 전략 2), Fig.2 5cm 확정(이미 가정), Fig.4 3.5cm(−0.05p). 잠재 −0.2p.
2. **[재가 불요] LaTeX 실측 보정**: 본 추정은 단어수 환산 — elsarticle 11pt 실조판에서 ±10% 변동 가능. Phase 7 1차 조판 후 재측정이 선행되어야 함.
3. **[재가 필요] §4 의무 서술 일부의 Appendix 구조 이동**: 예 — SMAP/MSL 경계조정 수치 상세(§A), DAGMM variant 주기(§A.1), threshold 관례 상술(§A). 각 directive의 "본문 명기" 요건 재해석이 필요하므로 integrator 단독 적용 불가.
4. **[재가 필요] 섹션 예산 재배분**: §4를 3.3→4.0p로 올리고 Appendix 일부 페이지와 상쇄하는 안 — PAGE_BUDGET가 단일 정본(ADV BLK-001)이므로 blueprint-reviser 경유 개정 필요.
5. **[플래그 유지] Table 2 landscape 지원 (RT V1)**: elsarticle sideways 본문 허용 여부 미검증 — 미지원 시 fallback 사다리 (a)→(b)→(c, V3 재결정 필수) 적용. 0.50p 가정이 깨지면 +0.2p 추가.

## 6. 잔존 이슈 (Phase 6/7 인계)

1. **분량 초과 +2.8p** — §5 위 사다리 적용 + 오케스트레이터 판단 필요 (본 보고의 최대 이슈).
2. **NUM sync group A/B** (registry §3): 데이터셋 수·베이스라인 수의 front placeholder ↔ §4 하드코딩 상수 동기화 — weakly-supervised 4종 미완료 시 "26→22" 연쇄 수정 목록이 registry에 명세됨.
3. **조건부 ablation row 5/6/7** (Table 3): 실험 미완료 시 §B.1 강등 + contribution bullet 3 표현 격하 (원고 §4.3에 conditional 괄호 잔존 — Phase 6에서 해소).
4. **§2.2 "to our knowledge" 스코핑**: 반증 후보(xue2022fewpositive, huang2022slavae) 최종 분석 후 재확인 (related_work drafter 노트 승계; D-008/C-011/C-025).
5. **각주 [^sd-fn]**: Phase 7 LaTeX 변환 시 \footnote{} 처리 + §3.4 인접 배치 재고 (related_work 노트 승계).
6. **TXT-001 (GPU 모델)**: 실험 metadata에서 채울 것 — 추정 기입 금지.
7. **§5 "approximately 50×"**: Appendix §B.3 wall-clock 실측과 괴리 시 표현 완화 (registry §5 조건).
8. **Acknowledgments·Appendix·References 본문**: v1 범위 외 — Phase 6/7에서 refs.bib 컴파일 및 Appendix(§A–C) 작성.
9. **미사용 bib key 5건** (§2 참조): Phase 6에서 인용 추가 여부 판단 (특히 xiong2020prenorm).
