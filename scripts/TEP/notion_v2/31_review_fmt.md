# Notion 포맷 검수 보고 — 20_draft.md

- **검수 대상**: `temp/0610/TEP/notion_v2/20_draft.md` (542줄, 테이블 9개, callout 13개)
- **대조 기준**: `.claude/agents/academic-writer.md` (Notion 포맷팅 규칙, SELF-VALIDATION 체크리스트), `.claude/agents/notion-expert.md` (TABLE FORMATTING CRITICAL, preserve 목록)
- **검수일**: 2026-06-11

## 체크리스트 결과

| # | 항목 | 기준 | 결과 |
|---|------|------|------|
| 1 | `fit-page-width="true"` | 모든 테이블 필수 | **PASS** — 9/9 테이블 (L25, 55, 89, 146, 211, 274, 327, 450, 493) |
| 2 | `header-row="true"` | 모든 테이블 필수 | **PASS** — 9/9 테이블 |
| 3 | 매트릭스 `header-column="true"` | 행 라벨 × 수치 cross-tab에 필수 | **PASS** — 매트릭스 6개 (L146, 211, 274, 327, 450, 493) 모두 적용. 미적용 3개 (L25, 55, 89)는 서술형 목록 테이블로 매트릭스 아님 |
| 4 | tr/td 멀티라인 + 탭 들여쓰기 | `<tr>`, `<td>`, `</tr>` 각각 별도 줄, 탭 들여쓰기 (인라인 시 행 병합 버그) | **PASS** — 인라인 `<tr><td>`/`</td><td>`/`</td></tr>` 패턴 0건. `cat -A` 실측으로 탭(`^I`) 들여쓰기 확인 (`<tr>`=1탭, `<td>`=2탭). 공백 들여쓰기 0건 |
| 5 | 셀 색상 형식 | `<td color="red">` 형식만 허용, `:green_bg[text]` 인라인 문법 금지 | **PASS** — `<td color>` 70건 전부 attribute 형식 (green 23 / orange 18 / red 29, 모두 문서화된 값). 인라인 `_bg[...]` 문법 0건 |
| 6 | Callout 문법 | `:::callout {icon="..." color="..._bg"}` — icon+color 필수, `:::` 닫기 | **PASS** — 13/13 callout에 icon+color 존재, 열기 13 = 닫기 13 균형 |
| 7 | Callout 색상 팔레트 | red_bg/blue_bg/green_bg/orange_bg/yellow_bg/purple_bg | **PASS** — blue_bg 4, green_bg 2, purple_bg 2, orange_bg 2, yellow_bg 2, red_bg 1. 팔레트 외 색상 0건 |
| 8 | 미문서화 속성 금지 | colspan/rowspan/style/align/width, `<th>`/`<thead>`/`<tbody>`/`<span>`/`<div>`/`<br>` 등 | **PASS** — 전부 0건 |
| 9 | H1 금지 | `# ` 헤딩 사용 금지 (H2부터) | **PASS** — H1 0건, 섹션은 모두 `##`/`###` |
| 10 | 인사이트 번호 연속성 | 연속 번호 1️⃣, 2️⃣... 건너뛰기 금지 + 개별 색상 | **PASS** — 1️⃣~5️⃣ (L404, 408, 412, 416, 420) 연속, 색상 5종 모두 상이 (blue/purple/orange/green/yellow). 판독 1~5 (L197, 201, 262, 319, 374)도 연속 |

## 위반 사항

**없음.** FMT 위반 0건. 검수 범위의 10개 항목 전부 PASS.

## 참고 (비위반 — 수정 불요)

- **NOTE-1** (L197): 판독 1 callout이 `purple_bg` 사용. 팔레트 용도 가이드상 분석=blue_bg, 비교조건=purple_bg인데, 해당 callout 내용이 micro vs macro **대조/비교**이므로 용도 적합으로 판단. 팔레트 내 색상이므로 위반 아님.
- **NOTE-2** (L173, 183): 최고치가 아닌 orange 셀에 볼드(`**+0.045**`, `**+0.093**`) 사용. 규칙은 "최고치 → green+볼드"를 명시할 뿐 그 외 볼드를 금지하지 않으며, 본문 L144에 "양수 G는 굵게 표시" 범례가 선언되어 있어 일관적. 위반 아님.
- **NOTE-3** (L473): 일반 `<td>` 내 `**...**` 볼드 — Notion 변환 시 정상 렌더링되는 문서화된 패턴. 위반 아님.

## VERDICT

```
RESULT: ACCEPT (formatting)
FMT_VIOLATIONS: 0
NOTES: 3 (informational, no action required)
```
