#!/usr/bin/env python3
"""Copy the results downloaded from audit/review.html into the audit workbook.

Each JSON file says which reviewer it belongs to; its answers go into that reviewer's sheet
(matched by item id), then the workbook is saved and the agreement and kappa summary is printed.

Usage:
  python scripts/audit_import.py audit/audit_v2.xlsx audit_reviewer_A.json audit_reviewer_B.json audit_reviewer_C.json
"""
import json, os, subprocess, sys
from openpyxl import load_workbook

CHECKS = ["answer_correct", "evidence_matches", "rationale_ok"]

book, files = sys.argv[1], sys.argv[2:]
wb = load_workbook(book)
for path in files:
    res = json.load(open(path, encoding="utf-8"))
    ws = wb[f"Reviewer {res['reviewer']}"]
    head = [c.value for c in ws[1]]
    col = {name: head.index(name) + 1 for name in CHECKS + ["Notes", "id"]}
    row_of = {ws.cell(row=r, column=col["id"]).value: r for r in range(2, ws.max_row + 1)}
    filled = 0
    for it in res["items"]:
        r = row_of.get(it["id"])
        if r is None:
            sys.exit(f"{path}: item {it['id']} is not in {book}")
        for c in CHECKS:
            if it.get(c):
                ws.cell(row=r, column=col[c], value=it[c])
        ws.cell(row=r, column=col["Notes"], value=it.get("notes") or None)
        filled += all(it.get(c) for c in CHECKS)
    print(f"{path}: reviewer {res['reviewer']}, {filled} of {len(res['items'])} items complete")
wb.save(book)
print(f"saved {book}\n")
subprocess.run([sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), "audit_kappa.py"), book])
