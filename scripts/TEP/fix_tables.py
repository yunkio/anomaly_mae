"""fix_tables.py — robustly fix table nesting + stray code fences in the Notion page.
- close each <table> after its LAST </tr> before the next <table> opens
- drop orphan </table> (close with no open table)
- drop unmatched ``` fences
Reports diffs so they can be applied to the published page too.
"""
import re
R = 'results/experiments/TEP_phase2_win100_ep30'
lines = open(f'{R}/notion_page.md').read().split('\n')

out = []
in_table = False
last_tr_idx = None        # index in `out` of last </tr> while in a table
issues = []
for ln in lines:
    s = ln.strip()
    if s.startswith('<table'):
        if in_table:        # previous table never closed → close it after its last </tr>
            if last_tr_idx is not None:
                out.insert(last_tr_idx + 1, '</table>')
                issues.append(f'INSERT </table> after last </tr> (line near: {out[last_tr_idx][:30]})')
            else:
                out.append('</table>')
        in_table = True; last_tr_idx = None
        out.append(ln); continue
    if s == '</table>':
        if not in_table:
            issues.append(f'DROP orphan </table>')
            continue       # drop orphan
        in_table = False; last_tr_idx = None
        out.append(ln); continue
    if s == '</tr>':
        out.append(ln); last_tr_idx = len(out) - 1; continue
    out.append(ln)
if in_table:
    if last_tr_idx is not None:
        out.insert(last_tr_idx + 1, '</table>'); issues.append('INSERT </table> at end (table left open)')
    else:
        out.append('</table>')

t = '\n'.join(out)
# unmatched code fences: count ```; if odd, drop the last lone one
fences = [i for i, l in enumerate(out) if l.strip().startswith('```')]
if len(fences) % 2 == 1:
    # drop the fence that has no partner: pair them greedily, the leftover is the issue
    # find a ``` immediately followed (next fence) by content that is NOT its close → simplest: drop a lone ``` that sits between a closed block and a heading
    # heuristic: drop the last ``` if the block before it is already balanced
    drop = fences[-1]
    issues.append(f'DROP unmatched ``` at out-line {drop}: {out[drop][:20]}')
    del out[drop]
    t = '\n'.join(out)

open(f'{R}/notion_page.md', 'w').write(t)
print('issues fixed:')
for i in issues: print('  -', i)
print(f'final: len {len(t)} | <table {t.count("<table")} </table> {t.count("</table>")} | ``` {t.count(chr(96)*3)}')
