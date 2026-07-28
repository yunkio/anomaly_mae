"""assemble_page.py — clean one-shot assembly of the 9 Notion sections."""
import json, re

F = '/tmp/claude-1000/-home-ykio-notebooks-claude/ac31b1b3-74c8-4adb-b0bd-00ce50e56d32/tasks/wnwe6ig4q.output'
R = 'results/experiments/TEP_phase2_win100_ep30'
d = json.load(open(F))
secs = d['result']['sections']

cleaned = []
for md in secs:
    m = re.search(r'(?m)^## ', md)            # strip polish-agent commentary before first '## '
    cleaned.append((md[m.start():] if m else md).rstrip())
t = '\n\n'.join(cleaned)

# 1) HTML entities -> Notion escapes (one backslash)
t = t.replace('&lt;', '\\<').replace('&gt;', '\\>').replace('&amp;', '\\&')

# 2) colspan row (Notion 미지원) -> remove, capture note
note = ''
m = re.search(r'<tr>\s*<td>pak \(headline metric\)</td>\s*<td colspan="8">(.*?)</td>\s*</tr>\s*', t, re.S)
if m:
    note = m.group(1).strip()
    t = t[:m.start()] + t[m.end():]

# 3) balance <table>/</table>: insert </table> wherever an open is followed by open w/o close; append if ends open
toks = [(mm.start(), mm.group()) for mm in re.finditer(r'<table|</table>', t)]
inserts = []
depth = 0
for pos, g in toks:
    if g == '<table':
        if depth >= 1:
            inserts.append(pos)        # need to close previous before this open
        depth = 1
    else:
        depth = max(0, depth - 1)
# apply inserts back-to-front (insert '</table>\n' on its own line before the offending open's line)
for pos in sorted(inserts, reverse=True):
    nl = t.rfind('\n', 0, pos)
    t = t[:nl] + '\n</table>' + t[nl:]
if t.count('<table') > t.count('</table>'):
    t = t.rstrip() + '\n</table>\n'

# 4) pak note -> paragraph right after the FIRST table (overview headline table)
if note:
    p = t.find('</table>')
    p = t.find('\n', p) + 1
    t = t[:p] + f'\n**pak (headline metric)**: {note}\n' + t[p:]

open(f'{R}/notion_page.md', 'w').write(t)
print('길이:', len(t), '| <table', t.count('<table'), '</table>', t.count('</table>'),
      '| colspan', t.count('colspan'), '| &lt;', t.count('&lt;'), '| inserts', len(inserts))
