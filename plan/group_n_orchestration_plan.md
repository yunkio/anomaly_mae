# Group N Orchestration Plan — Group M(269-278) 결과 분석 + Group N 신규 10개 설계

## 전체 목표
1. **Notion subpage** (35887856b207819db68ffca412ae1580) 구조 완벽 이해
2. **269-278 실험 결과 통계 분석 + 인사이트 도출** (raw data 직접 X)
3. Notion subpage에 분석 + 10개 신규 실험(Group N, 279-288) 추가
4. Notion main page (32887856b2078193819ccaec36207605) 결과 업데이트 (Rank 포함)

## Sub-Agent 팀 정의

| Agent | Type | Role |
|-------|------|------|
| **S0: Notion-Explorer** | notion-expert | (a) Group M subpage 구조 fetch+이해 (b) Main page 구조 이해 |
| **S1: Statistician** | statistician | 269-278 통계치 압축 추출 (성능, best epoch, loss 추이, GRL 상태, adaptive lambda) |
| **S2: DL-Analyst** | dl-analyst | (a) 개별 실험 분석 (b) 통계 분석 (c) 가설 수립 (d) 10개 실험 설계 |
| **S3: Critical-Reviewer** | critical-reviewer | 객관적 리뷰 (피드백 루프, 최소 2회) |
| **S4: Notion-Publisher** | notion-expert | 두 Notion 페이지 업데이트 |

## 데이터 범위
- 269-278 (10개) 중 현재 277 진행 중, 278 미시작
- 분석은 완료된 269-276 위주, 277/278은 부분 데이터로 보조 분석
- 277/278이 완료되면 후속 보강

## Group M 가설 매핑 (검증해야 할 내용)
| Exp | Hypothesis | Baseline | 검증 목표 |
|-----|-----------|----------|----------|
| 269 | H1 single-axis | 190 (GRL #2) | ep500이 anomaly_loss ON 환경에서도 작동? |
| 270 | H1+H3 | 208 (GRL #3) | fm_l2 + ep500 시너지? |
| 271 | H1+H3+H4 | 212 (GRL #4) | 모든 capacity 결합 시너지? |
| 272 | H2 trap | 190 + bal | balanced + anomaly_loss = catastrophic? |
| 273 | H1+H2 decomp | 265 (#1) - bal | balanced가 265의 필수? |
| 274 | H3 | 265 + fm_l2 | fm_l2가 cosine보다 우수? |
| 275 | H7 | 245 + ep500 | healthy GRL이 ep500에서 작동? |
| 276 | H8 | 254 + ep500 | w=0.5가 ep500에서 sweet? |
| 277 | H5 | 265 + wider_cls | wider classifier가 best baseline에서 rescue? |
| 278 | H6 (3-axis) | 247 + adapt_off + w=0.05 + ep500 | specialist → balance? |

## 실행 체크리스트

- [ ] **Phase 0**: 본 계획 파일 작성 (완료 후 체크)
- [ ] **Phase 1**: S0 Notion-Explorer — Group M subpage 구조 분석 → `plan/p1_subpage_structure.md`
- [ ] **Phase 2**: S1 Statistician — 269-278 통계 추출 → `temp/group_n_p2_stats.json` + `.md`
- [ ] **Phase 3**: S2 DL-Analyst (Part A) — 개별 실험 8-10개 분석 → `temp/group_n_p3_individual.md`
- [ ] **Phase 4**: S2 DL-Analyst (Part B) — 통계 + 가설 검증 + 10개 신규 설계 → `temp/group_n_p4_design.md`
- [ ] **Phase 5**: S3 Critical-Reviewer 1차 리뷰 → `temp/group_n_p5_review1.md`
- [ ] **Phase 6**: S2 1차 수정 → `temp/group_n_p6_revision1.md`
- [ ] **Phase 7**: S3 2차 리뷰 (ACCEPT/REJECT) → `temp/group_n_p7_review2.md`
- [ ] **Phase 8**: S4 Notion-Publisher (subpage) → Group N 추가
- [ ] **Phase 9**: S4 Notion-Publisher (main page) → Group L 결과 업데이트 + Group N 추가
- [ ] **Phase 10**: queue_exp279_288.json 생성 + 체인 추가

## 검증 기준
- 통계 추출 시 raw data 직접 읽기 금지 (압축 통계만 사용)
- 각 신규 실험: GRL ON 필수 (사용자 요구사항 유지)
- baseline은 RankAvg-verified
- 단일 축 isolation 우선

## 위험/주의사항
- 277/278이 아직 진행 중 → 부분 데이터로 분석, 완료 후 검증 필요
- Notion API: 페이지가 크므로 surgical updates 필요
- 274 (RankAvg #1, 11.8)가 새로운 success bar — Group N의 baseline 됨
