#!/usr/bin/env python3
"""Verify teacher outputs (step 7). Usage:
  python3 check_teacher.py BATCH.jsonl OUT.jsonl      -> prints one line per failing id, then a summary
Each OUT line: {"id", "answer", "evidence": [..], "rationale", "teacher_agrees": bool, "note"?}
Checks: schema; answer == gold (relaxed); every evidence string points at real content of the input;
rationale <= 40 words; teacher_agrees is true."""
import json, re, sys

def norm(s):
    s = str(s).lower().strip().rstrip(".")
    s = re.sub(r"[\"'`’*]", "", s)
    return re.sub(r"\s+", " ", s)

def num(s):
    s = str(s).strip().replace(",", "").replace("%", "").replace("$", "").replace("€", "").replace("£", "").rstrip(".").lstrip("*")
    try: return float(s)
    except ValueError: return None

def same_answer(a, g):
    if a is None or g is None: return a is None and g is None
    x, y = num(a), num(g)
    if x is not None and y is not None:
        return abs(x - y) <= max(1e-9, abs(y) * 0.05) if y != 0 else abs(x) < 1e-9
    return norm(a) == norm(g)

def table(text):
    rows = [l for l in text.split("\n") if l.startswith("|")]
    cells = [[c.strip() for c in r.strip().strip("|").split("|")] for r in rows]
    header, body = cells[0], [r for r in cells[2:]]
    return header, body

def cell_ok(val, cell):
    if norm(val) == norm(cell): return True
    a, b = num(val), num(cell)
    return a is not None and b is not None and abs(a - b) < 1e-6

EV_CELL = re.compile(r"^Row=(.+?), Col=(.+) \(([^()]*)\)$")
EV_COL = re.compile(r"^Col=(.+?) \(all rows\)$")
EV_ROW = re.compile(r"^Row=(.+?) \(all columns\)$")
EV_COLOUR = re.compile(r"^Colour: (.+?) = (.+)$")
EV_TITLE = re.compile(r"^Title: (.+)$")

def check_evidence(ev, text):
    header, body = table(text)
    H = [norm(h) for h in header]
    R = {norm(r[0]): r for r in body}
    m = EV_CELL.match(ev)
    if m:
        row, col, val = (x.strip() for x in m.groups())
        if norm(row) not in R: return f"row not in table: {row!r}"
        if norm(col) not in H: return f"column not in table: {col!r}"
        cell = R[norm(row)][H.index(norm(col))] if H.index(norm(col)) < len(R[norm(row)]) else ""
        if not cell_ok(val, cell): return f"value {val!r} != table cell {cell!r} at {row}/{col}"
        return None
    m = EV_COL.match(ev)
    if m:
        return None if norm(m.group(1)) in H else f"column not in table: {m.group(1)!r}"
    m = EV_ROW.match(ev)
    if m:
        return None if norm(m.group(1)) in R else f"row not in table: {m.group(1)!r}"
    m = EV_COLOUR.match(ev)
    if m:
        lab, col = norm(m.group(1)), norm(m.group(2))
        t = norm(text)
        if f"{lab} (colour: {col}" in t or f"{lab} = {col}" in t: return None
        return f"colour mapping not in input: {ev!r}"
    m = EV_TITLE.match(ev)
    if m:
        title = next((l[7:] for l in text.split("\n") if l.startswith("Title: ")), None)
        return None if title and norm(m.group(1)) in norm(title) else f"title text not in input: {ev!r}"
    return f"evidence not in an allowed form: {ev!r}"

def check(item, out):
    errs = []
    if not isinstance(out.get("evidence"), list) or not out["evidence"]: errs.append("evidence must be a non-empty list")
    if not isinstance(out.get("rationale"), str) or not out["rationale"].strip(): errs.append("rationale missing")
    elif len(out["rationale"].split()) > 40: errs.append(f"rationale has {len(out['rationale'].split())} words (max 40)")
    if out.get("teacher_agrees") is not True: errs.append("teacher_disputes_gold: " + str(out.get("note", "")))
    if not same_answer(out.get("answer"), item["gold_answer"]): errs.append(f"answer {out.get('answer')!r} != gold {item['gold_answer']!r}")
    for ev in out.get("evidence") or []:
        if not isinstance(ev, str): errs.append("evidence item not a string"); continue
        e = check_evidence(ev, item["input_text"])
        if e: errs.append(e)
    return errs

def main():
    items = {json.loads(l)["id"]: json.loads(l) for l in open(sys.argv[1])}
    outs = {}
    for i, l in enumerate(open(sys.argv[2])):
        if not l.strip(): continue
        try: o = json.loads(l)
        except Exception as e: print(f"line {i+1}: invalid JSON: {e}"); continue
        outs[o.get("id")] = o
    bad = 0
    for iid, it in items.items():
        if iid not in outs: print(f"{iid}: MISSING"); bad += 1; continue
        errs = check(it, outs[iid])
        if errs: print(f"{iid}: " + " | ".join(errs)); bad += 1
    print(f"SUMMARY: {len(items)} items, {len(items) - bad} pass, {bad} fail")

if __name__ == "__main__":
    main()
