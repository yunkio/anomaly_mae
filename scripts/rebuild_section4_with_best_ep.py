"""Rebuild Section 4 (PSM Leaderboard) tables — add best_ep in parentheses.

Strategy: parse existing insert_chunk_4.txt rows, look up best_ep per dataset from
notion_exp_results.json, and emit new table with format `score (best_ep)` for
cells that correspond to a single dataset. SMD aggregate / Avg / RA cells stay as-is.

Cell layout per row:
  0: Exp (e.g. "140" or "**274** 🥇")
  1: Name
  2: excl22  -> add best_ep
  3: A1      -> add best_ep
  4: A2      -> add best_ep
  5: SMD     -> aggregate, leave as-is
  6: PSM     -> add best_ep
  7: Avg(5DS) -> derived, leave as-is
  8: RA(60)   -> derived, leave as-is
"""
import json
import re
from pathlib import Path

ROOT = Path("/home/ykio/notebooks/claude")
DATA = json.loads((ROOT / "temp/notion_exp_results.json").read_text())
SRC = (ROOT / "temp/insert_chunk_4.txt").read_text()


def lookup_ep(exp_id_clean: str, ds_key: str) -> str:
    """Return best_ep string, or '' if missing."""
    rec = DATA.get(exp_id_clean)
    if not rec:
        return ""
    sub = rec.get(ds_key)
    if not isinstance(sub, dict):
        return ""
    ep = sub.get("best_ep")
    if ep is None:
        return ""
    return str(int(ep))


def clean_exp_id(raw: str) -> str:
    """Strip markdown / emoji to recover bare exp id (e.g., '**274** 🥇' -> '274')."""
    s = re.sub(r"\*+", "", raw)
    s = re.sub(r"[^\d]", "", s)
    return s


# Columns -> JSON key for best_ep lookup
COL_KEY = {
    2: "swat_excl22",
    3: "wadi_A1",
    4: "wadi_A2",
    6: "psm",
}


def rebuild_table(table_text: str) -> str:
    """Parse the <table> block and rewrite each data row with best_ep parens."""
    # Find rows
    out_lines = []
    in_row = False
    row_cells = []
    is_header = True

    for line in table_text.split("\n"):
        if line.strip() == "<tr>":
            in_row = True
            row_cells = []
            continue
        if line.strip() == "</tr>":
            in_row = False
            if is_header:
                out_lines.append("<tr>")
                for c in row_cells:
                    out_lines.append(f"<td>{c}</td>")
                out_lines.append("</tr>")
                is_header = False
                continue
            # Data row: row_cells[0]=Exp, [1]=Name, [2..8]
            exp_raw = row_cells[0]
            exp_id = clean_exp_id(exp_raw)
            new_cells = [exp_raw, row_cells[1]]
            for i in range(2, len(row_cells)):
                cell = row_cells[i]
                if i in COL_KEY:
                    ep = lookup_ep(exp_id, COL_KEY[i])
                    if ep and cell.strip() and cell.strip() not in ("—", "-"):
                        # Append (ep) — preserve markdown bold/etc inside cell
                        cell = f"{cell} ({ep})"
                new_cells.append(cell)
            out_lines.append("<tr>")
            for c in new_cells:
                out_lines.append(f"<td>{c}</td>")
            out_lines.append("</tr>")
            continue
        if in_row:
            # parse single <td>...</td> line
            m = re.match(r"<td>(.*)</td>$", line.strip())
            if m:
                row_cells.append(m.group(1))
            else:
                # multi-line cell or empty
                row_cells.append(line.strip())
            continue
        # passthrough
        out_lines.append(line)

    return "\n".join(out_lines)


def main():
    # The file has 2 tables: 4.1 and 4.2. Find their bounds.
    # Each table starts with <table> and ends with </table>.
    out = []
    i = 0
    while i < len(SRC):
        start = SRC.find("<table>", i)
        if start < 0:
            out.append(SRC[i:])
            break
        end = SRC.find("</table>", start)
        if end < 0:
            out.append(SRC[i:])
            break
        end += len("</table>")
        out.append(SRC[i:start])  # leading text
        table_text = SRC[start:end]
        rebuilt = rebuild_table(table_text)
        out.append(rebuilt)
        i = end

    new_text = "".join(out)
    (ROOT / "temp/insert_chunk_4_with_ep.txt").write_text(new_text)
    print(f"Wrote {ROOT / 'temp/insert_chunk_4_with_ep.txt'}")
    print(f"Original: {len(SRC)} chars, New: {len(new_text)} chars")


if __name__ == "__main__":
    main()
