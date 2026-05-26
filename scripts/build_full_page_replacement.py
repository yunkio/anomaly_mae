"""
Build the full corrected page content by:
1. Taking the most recent fetched Notion page
2. Reversing the placeholder corruption ($$SECTION_21_CONTENT$$)
3. Applying Section 2.1 replacement
4. Applying Section 2.2 replacement
5. (Optional) Adding SMD(15) to Section 4

Output: temp/notion_full_corrected.txt
"""
from pathlib import Path

SRC = Path("/home/ykio/.claude/projects/-home-ykio-notebooks-claude/2e95ca00-5a80-4852-9ddd-736c44003076/tool-results/mcp-claude_ai_Notion-notion-fetch-1779346145369.txt")
OUT = Path("/home/ykio/notebooks/claude/temp/notion_full_corrected.txt")

# Read the latest fetched page (JSON envelope)
import json
raw = SRC.read_text()
# Try parsing as JSON first
try:
    j = json.loads(raw)
    content = j.get("text", raw)
except json.JSONDecodeError:
    content = raw
content = content.replace("\\n", "\n").replace("\\t", "\t")

# Strip outer wrapper if present
# Look for "<content>\n" ... "\n</content>"
cs = content.find("<content>")
ce = content.rfind("</content>")
if cs != -1 and ce != -1:
    content = content[cs + len("<content>"):ce].strip()

# The page currently has the placeholder corruption.
# Replace $$SECTION_21_CONTENT$$ (with whatever escaping) + still-original 2.2 with my new content.

# Find positions
idx_2_header = content.find("# 2. 결과")
idx_22 = content.find("## 2.2 PAK_AUC_PRC")
idx_4 = content.find("# 4. PSM-Included")

# Section 2.1 spans from "# 2. 결과\n" to "## 2.2"
# Currently the page has: "# 2. 결과\n" + placeholder + "## 2.2 PAK_AUC_PRC..."
# We need to replace placeholder with the new Section 2.1 content (which starts with "## 2.1 ...")

# Read new content
new_21 = Path("/home/ykio/notebooks/claude/temp/notion_section_21.txt").read_text()
new_22 = Path("/home/ykio/notebooks/claude/temp/notion_section_22.txt").read_text()

# Replace from "# 2. 결과\n" up to "## 2.2" — replace placeholder with new 2.1
before_2 = content[:idx_2_header]
between = content[idx_2_header:idx_22]  # contains "# 2. 결과\n" + placeholder
after_22_start = content[idx_22:idx_4]  # current 2.2 (still broken with old single-col format)
after_4 = content[idx_4:]

# Build new content
new_between = "# 2. 결과\n" + new_21
new_22_section = new_22  # this includes trailing "---\n"

corrected = before_2 + new_between + new_22_section + after_4

OUT.write_text(corrected)
print(f"Old content: {len(content)} chars")
print(f"New content: {len(corrected)} chars")
print(f"Section 2.1 new: {len(new_21)} chars")
print(f"Section 2.2 new: {len(new_22)} chars")
