#!/usr/bin/env python3
"""Read a filled-in audit workbook and print, per check: % Yes per reviewer, agreement,
Fleiss' kappa (all reviewers) and Cohen's kappa for each pair of reviewers.

Usage: python scripts/audit_kappa.py audit/audit_v2.xlsx
"""
import collections, itertools, sys
from openpyxl import load_workbook

CHECKS = ["answer_correct", "evidence_matches", "rationale_ok"]


def col(ws, name):
    head = [c.value for c in ws[1]]
    i = head.index(name)
    return [r[i] for r in ws.iter_rows(min_row=2, values_only=True)]


def cohen(a, b):
    pairs = [(x, y) for x, y in zip(a, b) if x and y]
    n = len(pairs)
    if not n:
        return n, None, None
    po = sum(x == y for x, y in pairs) / n
    ca, cb = collections.Counter(x for x, _ in pairs), collections.Counter(y for _, y in pairs)
    pe = sum(ca[k] * cb[k] for k in set(ca) | set(cb)) / n / n
    return n, po, (po - pe) / (1 - pe) if pe < 1 else 1.0


def fleiss(columns):
    """Items rated by every reviewer only."""
    rows = [r for r in zip(*columns) if all(r)]
    n, m = len(rows), len(columns)
    if not n or m < 2:
        return n, None, None
    cats = sorted({v for r in rows for v in r})
    counts = [collections.Counter(r) for r in rows]
    p_i = [(sum(c[k] ** 2 for k in cats) - m) / (m * (m - 1)) for c in counts]
    p_bar = sum(p_i) / n
    p_j = [sum(c[k] for c in counts) / (n * m) for k in cats]
    pe = sum(p ** 2 for p in p_j)
    all_agree = sum(len(set(r)) == 1 for r in rows) / n
    return n, all_agree, (p_bar - pe) / (1 - pe) if pe < 1 else 1.0


fmt = lambda v, f: "n/a" if v is None else format(v, f)
wb = load_workbook(sys.argv[1])
names = [s[len("Reviewer "):] for s in wb.sheetnames if s.startswith("Reviewer ")]
for c in CHECKS:
    cols = {x: col(wb[f"Reviewer {x}"], c) for x in names}
    yes = "  ".join(f"{x} yes={fmt(sum(v == 'Yes' for v in cols[x] if v) / max(1, sum(1 for v in cols[x] if v)), '.0%')}" for x in names)
    n, agree, k = fleiss(list(cols.values()))
    print(f"{c:18} n={n:3}  {yes}  all agree={fmt(agree, '.0%')}  Fleiss kappa={fmt(k, '.2f')}")
    for x, y in itertools.combinations(names, 2):
        n2, po, k2 = cohen(cols[x], cols[y])
        print(f"{'':18}   {x}-{y}: agreement={fmt(po, '.0%')}  Cohen kappa={fmt(k2, '.2f')}")
