"""
Build separated Notion pages content from notion_full_corrected.txt.

Output files:
- temp/notion_main_page.txt — Section 1 only (Group A-N spec) + sub-page placeholders
- temp/notion_subpage_21.txt — Section 2.1 PAK_AUC_F1
- temp/notion_subpage_22.txt — Section 2.2 PAK_AUC_PRC
- temp/notion_subpage_4.txt — Section 4 PSM Leaderboard
"""
from pathlib import Path

SRC = Path("/home/ykio/notebooks/claude/temp/notion_full_corrected.txt")
OUT = Path("/home/ykio/notebooks/claude/temp")

content = SRC.read_text()

# Find boundaries
idx_section_2 = content.find("# 2. 결과")
idx_section_21 = content.find("## 2.1 PAK_AUC_F1")
idx_section_22 = content.find("## 2.2 PAK_AUC_PRC")
idx_section_4 = content.find("# 4. PSM-Included")

print(f"# 2. at: {idx_section_2}")
print(f"## 2.1 at: {idx_section_21}")
print(f"## 2.2 at: {idx_section_22}")
print(f"# 4. at: {idx_section_4}")

# Main page = everything from start up to "# 2. 결과"
main_page_content = content[:idx_section_2]
# Replace ending if needed
if not main_page_content.endswith("\n"):
    main_page_content += "\n"

# Add sub-page links section
main_page_content += """---
# 2. 결과 (Result Tables)
<callout icon="📊">
	**결과 테이블은 별도의 하위 페이지에 정리됨** (페이지 크기 관리 + 가독성).
	**총 165개 valid 실험 + Exp 285 placeholder** = 166 rows × 9 metric columns
	**컬럼 정의 (모든 결과 테이블 공통)**:
	- 셀: `pak_value (best_ep)`. SMD aggregates는 epoch 평균 처리 불가 — value만 표시.
	- **별표(*) 표시 컬럼**: Swat(excl22), Wadi A1, Wadi A2, SMD(15) — **RankAvg 4-DS 계산에 사용**.
	- **Avg(4*)** = 4 별표 DS의 값 평균
	- **RankAvg(4*)** = 4 별표 DS 각각의 등수의 평균 (낮을수록 좋음). 165 valid exp 기준
	- **SMD(15)**: TimeSeAD (Wagner et al., 2023) 권장 15-machine subset (excludes machine-1-1/1-3/1-4/1-5/1-6/1-8/2-5/2-8/3-4/3-5/3-7/3-10/3-11)
</callout>
__SUBPAGE_RESULTS__
"""

(OUT / "notion_main_page.txt").write_text(main_page_content)
print(f"\nMain page: {len(main_page_content)} chars → notion_main_page.txt")

# Single result subpage = Section 2.1 + 2.2 + Section 4
sub_results = content[idx_section_21:]
# Reformat headings (## → # for top-level on sub-page)
sub_results = sub_results.replace("## 2.1 PAK_AUC_F1 기준 결과", "# Section 2.1: PAK_AUC_F1 기준 결과 (Exp 119-285)", 1)
sub_results = sub_results.replace("## 2.2 PAK_AUC_PRC 기준 결과", "# Section 2.2: PAK_AUC_PRC 기준 결과 (Exp 119-285)", 1)
sub_results = sub_results.replace("# 4. PSM-Included Subset Leaderboard (60 models, 5-DS)",
                                    "# Section 4: PSM-Included Subset Leaderboard (60 models, 5-DS)", 1)

# Add a brief intro at top
intro = "Exp 119-285 모든 결과 정리 페이지 (PAK_AUC_F1, PAK_AUC_PRC, PSM Leaderboard).\n\n메인 페이지 callout에서 컬럼 정의 + TimeSeAD SMD(15) 정의 참조.\n\n---\n"
sub_results = intro + sub_results

(OUT / "notion_subpage_results.txt").write_text(sub_results)
print(f"Subpage Results (2.1 + 2.2 + 4): {len(sub_results)} chars → notion_subpage_results.txt")
