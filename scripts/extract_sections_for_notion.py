"""
Extract exact verbatim section content from fetched Notion page,
to use as old_str for notion-update-page replacements.
"""
import json
from pathlib import Path

SRC = Path("/home/ykio/.claude/projects/-home-ykio-notebooks-claude/2e95ca00-5a80-4852-9ddd-736c44003076/tool-results/mcp-claude_ai_Notion-notion-fetch-1779343996517.txt")
OUT_DIR = Path("/home/ykio/notebooks/claude/temp")

# The fetched content has literal \n and \t (escaped). Convert to real newlines.
content = SRC.read_text()
content = content.replace("\\n", "\n").replace("\\t", "\t")

# Find boundaries
def find_section(start_marker, end_marker, content):
    s = content.find(start_marker)
    e = content.find(end_marker, s) if end_marker else len(content)
    if s == -1: return None
    return content[s:e]

# Group N section (within Section 1) — ends before "# 2. 결과"
group_n_old = find_section("## Group N: Post-Group M", "# 2. 결과", content)
(OUT_DIR / "old_group_n.txt").write_text(group_n_old)
print(f"Group N old: {len(group_n_old)} chars")

# Section 2.1 — ends before "## 2.2"
section_21_old = find_section("## 2.1 PAK_AUC_F1", "## 2.2 PAK_AUC_PRC", content)
(OUT_DIR / "old_section_21.txt").write_text(section_21_old)
print(f"Section 2.1 old: {len(section_21_old)} chars")

# Section 2.2 — ends before "# 4." (note: there's a "---" separator before # 4.)
# Actually after 2.2 there's "---\n# 4." — so we trim trailing "---"?
section_22_old = find_section("## 2.2 PAK_AUC_PRC", "# 4. PSM-Included", content)
(OUT_DIR / "old_section_22.txt").write_text(section_22_old)
print(f"Section 2.2 old: {len(section_22_old)} chars")

# Section 4 - just for reference
section_4_old = find_section("# 4. PSM-Included", None, content)
(OUT_DIR / "old_section_4.txt").write_text(section_4_old)
print(f"Section 4 old: {len(section_4_old)} chars")
