> **[2026-06-13 종결]** 재진입 round 1 전 단계 완료: KBS 정합화 + Notion 단일 페이지 발행(https://www.notion.so/37e87856b20781fc92f6d8580c4b72a0). 본 문서는 당시 중단 스냅샷의 이력 기록이다.

---
phase: 7/8 재진입 (2026-06-13 완료 — 본 문서는 이력)
agent: orchestrator
directives: [M13]
last_modified: 2026-06-11
---

# RESUME STATE — 일시 중단 시점 스냅샷 (2026-06-11)

> **재개 절차**: 마스터 §0.8 재개 프로토콜 수행 (① 마스터 재정독 ② PHASE_LEDGER ③ TASK_BOARD ④ 본 문서) → 아래 "재개 시 실행 순서"를 위에서부터.

## 1. 배경 — 왜 재진입 중인가

Phase 0–8 완주·최종 게이트 PASS (commit 134e929) 후 사용자 추가 지시 2건:

1. **저널 확정 = Knowledge-Based Systems** → Phase 7 재진입 (KBS 포맷 정합화, D-015)
2. **R3 Notion 위치 정정 + 내용·구성 격상** (D-016 + 후속 지시): placeholder 명세는 **MAE for Anomaly Detection 페이지(31687856b20780e29fbcd961d69773ea) 하위**로, 내용은 "목적과 의도·목표·실제 실험 내용·설계·기대 결과"를 담아 **Notion만 보고 실험 설계·figure 작성 가능한 수준** + **Notion/마크다운 전문가 수준의 직관적 구성** + **완벽한 한국어 문장** — 내용 퀄리티와 가독성 어느 쪽도 희생 금지. 기존 발행본(비교실험 하위 37c87856b207810e83e3d1b5f14766fc)은 중립화 대상.

## 2. 완료된 것 (이번 재진입에서 — 전부 본 commit에 포함)

### KBS 포맷 정합화 (Phase 7 재진입) — 실질 완료
- `07_latex/KBS_FORMAT_REQUIREMENTS.md` — 공식 GFA 조사 (highlights 85자, 게재본 2단=5p, 선언 5종 필수, keywords ≤6, flat zip 등 — 출처 포함)
- 적용 8건 전부: highlights 5개 ≤85자 재작성(실측 79/81/84/84/83) + `highlights.txt` / keywords 7→6 ("Contaminated benchmark" 제거) / 선언 섹션 5종 (CRediT·COI·**GenAI 고지**·Data availability·Funding — 전부 [TO BE CONFIRMED BY AUTHORS]) / `\journal{Knowledge-Based Systems}` ×3 / **flat 구조** (sections/*.tex → 루트 git mv) / 재컴파일 3종 무오류 / **본문 8.997p 유지** (bit-identical 좌표) / zip 재패키징 (flat 13파일) + 단독 컴파일 PASS
- 기록: `PROSE_DIFF_LOG.md` §7, `pdf_qa/KBS_ADAPT_REPORT.md`
- **미니 감사 PASS 4/4** (`99_reviews/p7_kbs_miniaudit_r1.md`) — ai-phrasing/표절/method-truth/글자수. 비차단 NOTE 3건: ① "graceful decay"(H5) vs 본문 "degradation" 용어 변이(85자 제약상 허용) ② NUM-003 해소 시 highlights.txt 동시 갱신 의무 ③ Data availability 기입 시 SWaT/WaDi는 iTrust 신청 기반 — "publicly available" 문구 정밀화 (→ 핸드오프 노트 등재 예정)

### Notion 확장판 (Phase 8 재진입) — 콘텐츠 작성 완료, 정제 직전에 중단
- `08_final_audit/NOTION_ENRICHED_B1_body.md` — 본문 FIG 4종(FIG-1/2/3/4)+TAB 3종(+TAB-4 흡수 기재) 7페이지, 연결 NUM 28건 배속. 전 페이지에 🎯목적·의도(블루프린트 §12/§14/§15 연결) + 🏁목표·기대 결과(성공 기준 + 다른 패턴 시 해석) 신규 차원 — r2 실행 지침 전수 계승, 수치 창작 0
- `08_final_audit/NOTION_ENRICHED_B2_appendix.md` — Appendix 표 8종(묶음 2)+TAB-B1~B4+FIG-B1+ALG-C1+TXT+R-PROBE 10페이지 + **OVERVIEW(부모 페이지 대시보드 자료)**. 주의: TAB-B1 = contaminated-training(무절제) 비교가 정본 (standard-split은 TAB-2 하단 블록 소관 — B2가 지시문 오기를 정본 기준으로 정정)

## 3. 재개 시 실행 순서 (이 순서 그대로)

1. **[C] Notion 정제 agent dispatch** ← **중단 지점 (dispatch 직전 거부됨, 산출물 없음)**
   - 임무: B1+B2 → `08_final_audit/NOTION_FINAL_PAGES.md` 통합 정제. ① 페이지 순서: OVERVIEW → FIG 1–4 → TAB 1–3 → Appendix(A 묶음, B1–B4, FIG-B1) → ALG → TXT → R-PROBE (`<!-- PAGE: {ID} -->` 경계 유지) ② 구성 통일: 템플릿 차원·이모지(💡🎯🏁🧪📊📝⚠️🔢) 통일 + 페이지별 메타 표(유형/소스 분류/우선순위/의존성) + OVERVIEW를 "오늘 무엇부터" 30초 대시보드로(무학습 즉시 가능/GPU 신규/대기 3구역) ③ 한국어 전수 정제(번역투·모호어 금지, 용어 통일+영문 병기) ④ 렌더 안전(헤딩/표/인용/코드/구분선/이모지/bold만) ⑤ 무손실 기계 검증(FIG 5·TAB 12·ALG 1·NUM 31·TXT 2·R-PROBE + 영문 캡션 수)
2. **[D] 독립 검수 agent**: NOTION_FINAL_PAGES ↔ r2 spec 사실 보존 + "페이지만 보고 실험 설계 가능" 실행성 + 한국어 가독성 + 렌더 안전 — BLOCKER/MAJOR 0까지 수정 루프
3. **발행** (notion-expert): **부모+하위 페이지 구조** — 부모 "📋 논문 Placeholder 실험·Figure 상세 명세" under `31687856b20780e29fbcd961d69773ea` + `<!-- PAGE -->` 단위 하위 페이지 (~19장). create-pages 호출 (페이지 배열 — 긴 콘텐츠 update 계열 금지). 발행 후 re-fetch 렌더링 검증 (부모 + 하위 표본 4장 이상)
4. **구 페이지 중립화**: `37c87856b207810e83e3d1b5f14766fc` 제목을 "↪ [이관됨] …"로 변경 + 본문 최상단에 새 페이지 링크 1줄 (update-page 1줄은 안전)
5. **마감**: COVERAGE_MATRIX R3 근거 → 새 URL / T7·R6 근거에 KBS 정합 추가 / PHASE_LEDGER Phase 7 RE-ENTRY→DONE + Phase 8 재진입 round 기록·종결 / TASK_BOARD 재진입 절 갱신 / phase8_report에 KBS·Notion 재작업 부록 + 저자 액션 추가 (선언 5종 확정 — 특히 GenAI 고지 문구, highlights.txt 별도 제출, iTrust 데이터 문구, NUM-003↔highlights.txt 동기화) / **git commit** / 사용자 최종 보고

## 4. 산출물 위치 빠른 참조

| 항목 | 경로 |
|------|------|
| KBS 규정 | `paper/07_latex/KBS_FORMAT_REQUIREMENTS.md` |
| KBS 적용 보고 | `paper/07_latex/pdf_qa/KBS_ADAPT_REPORT.md` |
| KBS 미니 감사 | `paper/99_reviews/p7_kbs_miniaudit_r1.md` |
| 최신 zip (flat 13파일, KBS 반영) | `paper/07_latex/overleaf_package.zip` |
| 확장판 원고 B1/B2 | `paper/08_final_audit/NOTION_ENRICHED_B{1,2}_*.md` |
| 관련 결정 | DECISION_LOG D-015 (KBS), D-016 (Notion 위치·격상) |
