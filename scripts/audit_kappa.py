#!/usr/bin/env python3
"""Read a filled-in audit workbook and print agreement and Cohen's kappa per check.

Usage: python scripts/audit_kappa.py audit/audit_v2.xlsx
"""
import collections, sys
from openpyxl import load_workbook

CHECKS = ["answer_correct", "evidence_matches", "rationale_ok"]


def col(ws, name):
    head = [c.value for c in ws[1]]
    i = head.index(name)
    return [r[i] for r in ws.iter_rows(min_row=2, values_only=True)]


def kappa(a, b):
    pairs = [(x, y) for x, y in zip(a, b) if x and y]
    n = len(pairs)
    if not n:
        return n, None, None
    po = sum(x == y for x, y in pairs) / n
    ca, cb = collections.Counter(x for x, _ in pairs), collections.Counter(y for _, y in pairs)
    pe = sum(ca[k] * cb[k] for k in set(ca) | set(cb)) / n / n
    return n, po, (po - pe) / (1 - pe) if pe < 1 else 1.0


wb = load_workbook(sys.argv[1])
A, B = wb["Reviewer A"], wb["Reviewer B"]
for c in CHECKS:
    a, b = col(A, c), col(B, c)
    n, po, k = kappa(a, b)
    yes_a = sum(x == "Yes" for x in a if x) / max(1, sum(1 for x in a if x))
    yes_b = sum(x == "Yes" for x in b if x) / max(1, sum(1 for x in b if x))
    print(f"{c:18} n={n:3}  A yes={yes_a:.0%}  B yes={yes_b:.0%}  agreement={po if po is None else f'{po:.0%}'}  kappa={k if k is None else f'{k:.2f}'}")
