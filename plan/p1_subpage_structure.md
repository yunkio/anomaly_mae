---
agent: notion-expert
phase: 1 (structure mapping)
status: COMPLETE
timestamp: 2026-05-11
purpose: Document structure of two Notion pages so downstream agents can perform surgical edits to add Group N (Exp 279-288)
---

# Notion Page Structure Maps

Two pages analyzed:
- **Subpage** `35887856b207819db68ffca412ae1580` — Group H-L analysis + Group M (269-278) design (~51K chars)
- **Main page** `32887856b2078193819ccaec36207605` — Top-level tracker for Exp 119-278 (~74K chars)

---

## Subpage (35887856b207819db68ffca412ae1580) Map

**Title**: `📊 Group H-L 종합 분석 + Group M (Exp 269-278) 설계`
**Parent**: `MAE for Anomaly Detection` (31687856b20780e29fbcd961d69773ea)
**Icon**: 📊

### Top of page (before any heading)
- **Callout 1** `icon="🎯" color="blue_bg"` — **Executive Summary** (Top performers, 핵심 발견, 신규 실험, Baseline 성공 기준)
- **Callout 2** `icon="🔄" color="purple_bg"` — **v3 Update (2026-05-07)**: GRL-focused redesign rationale
- **Callout 3** `icon="⚠️" color="red_bg"` — **분석 메트릭 정정 (Updated 2026-05-02)** — RankAvg correction details
- `---` divider

### `## 1. 개별 실험 분석 (Phase 2A — Group H-L)`
Per-group analysis. Each group:
- `### Group X (Exp range) — descriptor`
- **Callout** `icon="🧪" color="gray_bg"` — Group Overview
- `#### Key Experiments (verbatim)` — bullet/bold descriptions of selected experiments
- `#### Group X — 나머지 ablation 요약` — compact summary table (`fit-page-width="true"`)

Subsections (all under `## 1`):
- `### Group H (Exp 185-217) — sd=2 + GRL w=0.2 ablation` (Key Exp + remainder table)
- `### Group I (Exp 218-239) — GRL OFF, sd/td/fm/freeze 조합` (Key Exp + remainder table)
- `### Group J (Exp 240-257) — GRL Performance Optimization` (Key Exp + remainder table)
- `### Group K (Exp 258-263) — Inference & Freeze Ablation` (callout only, no key-exp section)
- `### Group L (Exp 264-267) — exp247 Extension` (callout only)

### `## 2. 통계 분석 (Phase 2B — 83개 실험 집계)`
- `### 2.1 Best Epoch Distribution by Dataset (A.1)` + table (Dataset/Pre-warmup/Post-warmup/Total/%Post)
- `### 2.2 GRL Collapse Patterns (A.2)` + table (Group/N/healthy/near_random/collapsed_anom/%healthy)
- `### 2.3 Single-Variable Causal Isolation (A.5)` + table (Variable/A→B/Δexcl22/ΔA1/ΔA2/ΔSMD/해석)
- `### 2.4 GRL Aggregate Penalty (A.7) — Top-10 그룹 구성` + table (Group/GRL/Mean excl22/vs J)
- `### 2.5 A2 W-Cluster Analysis (A.8)` + table (Cluster/A2 W value/조건/검증된 예시)
- `### 2.6 A1 Champion Equivalence (A.9, MI2)` + table (Rank/Exp/A1 best ep/Config)

### `## 3. 가설 (Hypotheses H1-H8)`
- Table: H#/가설/Confidence/핵심 Evidence (8 rows, all GRL-focused hypotheses)

### `## 4. Group M 신규 10개 실험 (Exp 269-278) — v3 GRL-focused`
- **Callout** `icon="🎯" color="blue_bg"` — v3 design principles + H1-H8 summary
- `### 4.1 Compact 실험 설계 테이블 (v3, all GRL ON)` — table: Exp/Name/Hypothesis/Baseline/Override (10 rows)
- `### 4.2 Coverage Matrix (v3, P=primary, S=secondary)` — table: Exp/Name/H1..H8/Baseline RankAvg/ep (10 rows)
- `### 4.3 Key Experiments Detail (v3 GRL-focused)` — 10 subsections `#### Exp 269..278: ...`
- `### 4.4 Code Change Required (276, 277)` + callout `icon="⚠️" color="red_bg"`

### `## 5. 실행 계획`
- **Callout** `icon="⏱️" color="orange_bg"` — Queue file, chain, 예상 소요, Success bar
- `### Practical Recommendations (v3 GRL-focused)` — bullets organized by hypothesis priority
- **Callout** `icon="📝" color="gray_bg"` — 참고 자료 (source file list)
- `---` divider

### `## 6. Corrections Log (2026-05-02 RankAvg Revision)`
- **Callout** `icon="📜" color="yellow_bg"` — log description
- Table: # / 이전 주장 (잘못됨) / RankAvg 관점 정정
- `### Group M 교체 요약 (4 실험)` + table: Exp/이전 설계/교체 후/교체 이유
- `### 권고 액션` — numbered list (재평가 필요 / config 검증 / 메트릭 표준화)

### INSERTION POINTS for Group N (Exp 279-288)

The user must decide whether to:
- (a) **Add a new sibling subpage** for Group N analysis (cleaner; current page already focuses on H-L + M design)
- (b) **Append Group N section to this subpage** (riskier, increases length)

If (b) chosen, insertion strategy:
1. **Group N design section** — insert as **new `## 7. Group N 신규 10개 실험 (Exp 279-288)`** AFTER `### 권고 액션` (end-of-page append). Use `update_content` with `old_str` = the last line `"3. **메트릭 표준화**: 향후 모든 리포트는 RankAvg를 primary, Avg pak_f1을 secondary annotation으로 사용 권장."` and append below.
2. **Group M results back-fill** — once 269-278 results land, add a `### 4.5 Group M 결과 요약` AFTER `### 4.4 Code Change Required (276, 277)` and BEFORE `---\n## 5. 실행 계획`. Anchor: `"### 4.4 Code Change Required (276, 277)"` block end then `"---\n## 5. 실행 계획"`.
3. **Title update** — change `📊 Group H-L 종합 분석 + Group M (Exp 269-278) 설계` → `📊 Group H-M 종합 분석 + Group N (Exp 279-288) 설계` via `update_properties` (only if Group N appended here).
4. **Executive Summary callout** — needs revision for new findings; replace **Callout 1** content via `update_content`.

### ANCHOR STRINGS (verbatim, escape-free fragments for `old_str`)
Use these as starting points for surgical edits. Each is unique in the page.
- `## 1. 개별 실험 분석 (Phase 2A — Group H-L)`
- `### Group L (Exp 264-267) — exp247 Extension`
- `## 2. 통계 분석 (Phase 2B — 83개 실험 집계)`
- `## 3. 가설 (Hypotheses H1-H8)`
- `## 4. Group M 신규 10개 실험 (Exp 269-278) — v3 GRL-focused`
- `### 4.1 Compact 실험 설계 테이블 (v3, all GRL ON)`
- `### 4.2 Coverage Matrix (v3, P=primary, S=secondary)`
- `### 4.3 Key Experiments Detail (v3 GRL-focused)`
- `### 4.4 Code Change Required (276, 277)`
- `## 5. 실행 계획`
- `### Practical Recommendations (v3 GRL-focused)`
- `## 6. Corrections Log (2026-05-02 RankAvg Revision)`
- `### Group M 교체 요약 (4 실험)`
- `### 권고 액션`
- Last line (end-of-page anchor): `3. **메트릭 표준화**: 향후 모든 리포트는 RankAvg를 primary, Avg pak_f1을 secondary annotation으로 사용 권장.`

### V2 LEGACY REMNANTS (cleanup candidates)
- Callout 2 `icon="🔄" color="purple_bg"` references "이전 v2 권고는 자동 폐기" — keep (provides context, harmless).
- "v3에서 제거된 이전 v2 추천" sub-bullets in `### Practical Recommendations` — can remain (historical record).
- No structural duplication issues found. Page is internally consistent at v3.

---

## Main Page (32887856b2078193819ccaec36207605) Map

**Title**: `🧪 Exp 119-278: Mechanism, Depth, Epoch & Optimal Config Ablation`
**Parent**: `MAE for Anomaly Detection` (31687856b20780e29fbcd961d69773ea)
**Icon**: 🧪

### Top of page (before any heading)
- **Callout 1** `icon="📊" color="blue_bg"` — **실험 119-164 구성 정리** — Mechanism/Depth/Epoch/Optimal Config Ablation summary listing 12 groups A→L
- `---` divider

### `# 1. Experiment Groups`
- **Callout 2** `icon="🎯" color="yellow_bg"` — short orientation (5 groups A→E, ablation chain)

Per-group config sections (12 groups total). Each follows the pattern:
- `## Group X: name (Exp range, epoch, offset state)`
- **Callout** with icon — Group Base / 목적 / experimental notes
- **Config Table** `fit-page-width="true"` — columns vary by group, typically `Exp | Name | (Base) | Override | (Baseline) | 목적 / 검증 요소`

Group-by-group breakdown:
1. `## Group A: Mechanism Ablation (Exp 119-133, 50ep, offset=False)` — callout `🔧 gray_bg` + table
2. `## Group B: FM+OD 기반 Depth/Epoch (Exp 134-139)` — callout `📐 green_bg` + table
3. `## Group C: Epoch Scaling + GRL×Depth (Exp 140-148, 100ep)` — callout `🚀 blue_bg` + table
4. `## Group D: Offset=True 보정 (Exp 149-152)` — callout `🔄 orange_bg` + table
5. `## Group E: 최적 조합 탐색 (Exp 153-162, 200ep+, offset=True)` — callout `⭐ yellow_bg` + table
6. `## Group F: GRL 검증 (Exp 165-172, 200ep, offset=True)` — callout `🔬 purple_bg` + table
7. `## Group G: GRL 개선 + 추가 검증 (Exp 173-184, 200ep, offset=True)` — callout `🧪 red_bg` + table
8. `## Group H: GRL 최적 조합 (Exp 185-189, 200ep, offset=True)` — callout `🔗 green_bg` + table (note: section header says 185-189 but page contains larger config table covering H-range)
9. `## Group I: sd=1 Loss/Architecture 조합 탐색 (Exp 218-227, 200ep+, offset=True)` — callout `⭐ yellow_bg` + table
10. `## Group J: GRL Performance Optimization (Exp 240-257, 200-300ep, offset=True)` — has sub-heading `### Baseline & Verification Matrix` + callout `🎯 red_bg` + table
11. `## Group K: Inference & Freeze Ablation (Exp 258-263, 200ep, offset=True)` — callout `🔬 purple_bg` + table (6 rows: 258-263)
12. `## Group L: exp247 Extension + GRL-focused v3 (Exp 264-278)` — callout `📈 green_bg` + table (15 rows: 264-268 + 269-278)

### `# 2. 결과`
- **Callout** `icon="📊" color="blue_bg"` — describes table semantics (`PAK_AUC_F1 기준 전체 결과 (Exp 119-246)`, Avg/RankAvg definitions, completion notes)
- **Single master result table** `fit-page-width="true" header-row="true" header-column="false"`
  - Columns: `Exp | Name | SWaT (full) | SWaT (excl22) | WaDi A1 | WaDi A2 | SMD | Avg | Rank Avg`
  - Group divider rows: bold cells with `<br>` (e.g. `**Group A: Mechanism Ablation (119-133, 50ep, offset=False)<br>base : 119**`) followed by experiment rows
  - Current group dividers present: Group A → Group L (all 12 groups). Group L rows include 264-278 with all `—` placeholders.
  - **Current last row**: `<tr><td>278</td><td>247_adapt_off_w005_ep500</td><td>—</td>...<td>—</td></tr>` — last `</tr>` before `</table>`.

### INSERTION POINTS for Group N (Exp 279-288)

#### A. Title Update
- Use `update_properties` with `properties.title` = `🧪 Exp 119-288: Mechanism, Depth, Epoch & Optimal Config Ablation`
- Current title token to swap: `Exp 119-278` → `Exp 119-288`

#### B. Callout 1 (top summary)
- Add Group N to the 12-group enumeration. Anchor: the full callout block starting `<callout icon=\"📊\" color=\"blue_bg\">` (very first callout). Replace the `12개 그룹` line; extend to "13개 그룹" with `... → **M**(269-278, ...) → **N**(279-288, ...)`.
- Note: existing text currently says "10개 그룹" then enumerates 12 groups (A through L). User should decide whether to fix that mismatch (legacy stale count) while adding Group N — recommend updating to "12개 그룹" plus M and N if M is folded in, or "14개 그룹" if both are added explicitly.

#### C. New Group N config section
- Insert AFTER the `## Group L:` table closing `</table>` and BEFORE `# 2. 결과`.
- Anchor pair:
  - `old_str` end-of-Group-L marker: `</table>\n# 2. 결과` (preceded by Group L's final `<tr>` for row 278 — choose a unique snippet such as `<td>278</td>\n<td>247_adapt_off_w005_ep500</td>\n<td>247</td>\n<td>\`grl_adaptive_lambda=False, grl_loss_weight=0.05, num_epochs=500, warmup=250\`</td>\n<td>247 (RankAvg #5 GRL) + 248 mech</td>\n<td>H6 specialist→balance (3-axis). 248 A2 specialist mechanism + 247 base + ep500</td>\n</tr>\n</table>\n# 2. 결과`).
  - Or simpler: locate `\n# 2. 결과\n<callout icon=\"📊\" color=\"blue_bg\">` and insert new `## Group N: ...` section + callout + table immediately before it.
- Recommended structure (mirroring Group L):
  ```
  ## Group N: name (Exp 279-288, ...epoch..., offset=True)
  <callout icon="..." color="..._bg">
    Base / 목적 / 변경 요약
  </callout>
  <table fit-page-width="true">
    <tr><td>Exp</td><td>Name</td><td>Base</td><td>Override</td><td>Baseline</td><td>검증 요소</td></tr>
    ... 10 rows for 279-288 ...
  </table>
  ```

#### D. Result table — Group N rows
- Insert AFTER the last `</tr>` (Exp 278 row) and BEFORE the closing `</table>` of the master result table.
- Anchor: the exact tail block (verbatim):
  ```
  <tr>
  <td>278</td>
  <td>247_adapt_off_w005_ep500</td>
  <td>—</td>
  <td>—</td>
  <td>—</td>
  <td>—</td>
  <td>—</td>
  <td>—</td>
  <td>—</td>
  </tr>
  </table>
  ```
- Replacement: insert a Group N divider row + 10 placeholder rows, keeping `</table>` at the end.
  ```
  <tr>
  <td>278</td>
  ... (unchanged) ...
  </tr>
  <tr>
  <td>**Group N: name (279-288, ...)**</td>
  <td></td>... (8 empty cells) ...
  </tr>
  <tr><td>279</td><td>name</td><td>—</td>...<td>—</td></tr>
  ... (10 rows)
  </table>
  ```

#### E. Group M / L back-fill (optional, when 269-278 results arrive)
- Replace `—` cells in rows 264-278 of the master result table. Each row is uniquely identified by `<td>{N}</td>\n<td>{name}</td>` pair.
- Anchor pattern per row (verbatim): `<tr>\n<td>269</td>\n<td>190_ep500</td>\n<td>—</td>\n<td>—</td>\n<td>—</td>\n<td>—</td>\n<td>—</td>\n<td>—</td>\n<td>—</td>\n</tr>` → fill in metric values.
- Callout 1 at top should update `Exp 119-246` reference to `Exp 119-278` (or higher) when results land.

### ANCHOR STRINGS (verbatim from page, all unique)
Section headers:
- `# 1. Experiment Groups`
- `## Group A: Mechanism Ablation (Exp 119-133, 50ep, offset=False)`
- `## Group B: FM+OD 기반 Depth/Epoch (Exp 134-139)`
- `## Group C: Epoch Scaling + GRL×Depth (Exp 140-148, 100ep)`
- `## Group D: Offset=True 보정 (Exp 149-152)`
- `## Group E: 최적 조합 탐색 (Exp 153-162, 200ep+, offset=True)`
- `## Group F: GRL 검증 (Exp 165-172, 200ep, offset=True)`
- `## Group G: GRL 개선 + 추가 검증 (Exp 173-184, 200ep, offset=True)`
- `## Group H: GRL 최적 조합 (Exp 185-189, 200ep, offset=True)`
- `## Group I: sd=1 Loss/Architecture 조합 탐색 (Exp 218-227, 200ep+, offset=True)`
- `## Group J: GRL Performance Optimization (Exp 240-257, 200-300ep, offset=True)`
- `### Baseline & Verification Matrix`
- `## Group K: Inference & Freeze Ablation (Exp 258-263, 200ep, offset=True)`
- `## Group L: exp247 Extension + GRL-focused v3 (Exp 264-278)`
- `# 2. 결과`

Content anchors:
- Callout 1 opening: `<callout icon="📊" color="blue_bg">\n\t**실험 119-164 구성 정리** — Mechanism, Depth, Epoch Scaling & Optimal Config Ablation`
- 12-group enumeration line: `10개 그룹: **A**(119-133, Mechanism) → **B**(134-139, Depth)` (legacy stale "10개" prefix; bug)
- Result table opening: `<table fit-page-width="true" header-row="true" header-column="false">\n<tr>\n<td>Exp</td>\n<td>Name</td>\n<td>SWaT (full)</td>`
- Group L divider row in result table: `<td>**Group L: exp247 Extension + GRL-focused v3 (264-278)**</td>`
- Last data row (Exp 278): `<td>278</td>\n<td>247_adapt_off_w005_ep500</td>\n<td>—</td>\n<td>—</td>\n<td>—</td>\n<td>—</td>\n<td>—</td>\n<td>—</td>\n<td>—</td>\n</tr>\n</table>`

### V2 LEGACY REMNANTS / Inconsistencies
1. **`10개 그룹` count is stale** — enumeration lists 12 groups (A through L). Should be updated to reflect actual count when Group N lands ("13개 그룹" or "14개 그룹").
2. **`PAK_AUC_F1 기준 전체 결과 (Exp 119-246)`** in Section 2 callout — out of date; current rows extend to 278 (placeholders). Should be updated whenever results table grows.
3. **`247-257: 미실행 (247 진행중)`** note in same callout — likely outdated as Group J (240-257) and Group K/L have been queued/run. Worth verifying & rewriting when Group N is added.
4. **Group H section header** says `Exp 185-189` but Group H actually spans 185-217 per subpage. Stale numbering — config table on page covers wider range. Note for cleanup pass.
5. **Group L title** correctly says `(Exp 264-278)` — already extended for v3 redesign. Good.
6. Tables use multi-line `<tr>` / `<td>` format consistently (Notion-compatible). No table-merge bug.

---

## Cross-page Consistency Notes for Group N
- **Group N hypotheses** (likely H9-Hxx): should be designed in the **subpage** (Section 7 if appended, or a new sibling) with full justification.
- **Group N config rows**: must appear in **main page** Group N section + master result table.
- **Subpage Executive Summary callout** needs revision to reference Group N's expected outcomes.
- **Main page Callout 1** group enumeration line and `Exp 119-XXX` references need updates.
- **Title fields** on both pages should reflect the new range (`119-288`, `Group H-M + Group N`).

## File Size
This map is approximately 11 KB — well under the 20 KB target.
