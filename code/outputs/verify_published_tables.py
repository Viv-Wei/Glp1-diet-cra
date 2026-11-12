import sys
from openpyxl import load_workbook

GEN, PUB = "data/tables/Supplementary_Tables_v5.xlsx", "data/tables/Supplementary_Tables_published.xlsx"

def rows(wb, s):
    r = list(wb[s].iter_rows(values_only=True))
    while r and all(v is None for v in r[-1]): r.pop()
    return r

g, p = load_workbook(GEN, read_only=True), load_workbook(PUB, read_only=True)
if g.sheetnames != p.sheetnames: sys.exit(f"Different tables: {set(g.sheetnames) ^ set(p.sheetnames)}")
bad = 0
for s in p.sheetnames:
    a, b = rows(g, s), rows(p, s)
    d = [(i + 1, j + 1) for i in range(max(len(a), len(b))) for j in range(max(len(a[i]) if i < len(a) else 0, len(b[i]) if i < len(b) else 0))
         if (a[i][j] if i < len(a) and j < len(a[i]) else None) != (b[i][j] if i < len(b) and j < len(b[i]) else None)]
    if d: bad += 1; print(f"{s}: {len(d)} cells differ, first at row {d[0][0]}, column {d[0][1]}")
print(f"{len(p.sheetnames) - bad} of {len(p.sheetnames)} sheets identical to the published workbook")
sys.exit(1 if bad else 0)
