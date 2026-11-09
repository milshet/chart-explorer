"""Compare blind labels with the real answers (run only AFTER labelling).

Final confidence:
  high   = blind answer matches gold (numbers within 5%, text same after normalising)
  medium = numbers within 5-10%, or match but labeller said low confidence
  low    = mismatch -> left out of training
"""
import json, sys, re, pathlib, collections
out = pathlib.Path(sys.argv[1])
key = {k["pack_id"]: k for k in map(json.loads, open(out / "key" / "answers.jsonl", encoding="utf-8"))}
labs = [json.loads(l) for f in sorted((out / "outputs").glob("*.jsonl")) for l in open(f, encoding="utf-8") if l.strip()]
missing = sorted(set(key) - {l["pack_id"] for l in labs})
if missing: print(f"WARNING: {len(missing)} packs have no label yet, e.g. {missing[:5]}")

def num(x):
    if isinstance(x, (int, float)): return float(x)
    if isinstance(x, str):
        m = re.fullmatch(r"\s*[-+]?[\d,]*\.?\d+\s*%?\s*", x)
        if m: return float(x.replace(",", "").replace("%", ""))
    return None
def norm(x): return re.sub(r"[^a-z0-9.]+", " ", re.sub(r"['\u2019]", "", str(x).lower())).strip()
def parts(x):
    """Split list-like answers: "[A, B]", "A and B", "A, B" -> sorted normalised items."""
    s = str(x).strip().strip("[]")
    return sorted(p for p in (norm(t) for t in re.split(r"\s*,\s*|\s+and\s+|\s*;\s*", s)) if p)

ORDER = ["pack_id", "evidence", "rationale", "answer_type", "unit", "answer", "confidence"]
rows, stats = [], collections.Counter()
for l in labs:
    k = key[l["pack_id"]]; g, b = k["answer"], l.get("answer")
    stats["key_order_ok"] += list(l) == ORDER
    if isinstance(b, list): b = ", ".join(map(str, b))
    if k["answer_type"] == "unanswerable" or g is None:
        m = "exact" if b is None else "miss"
    else:
        gn, bn = num(g), num(b)
        if gn is not None and bn is not None:
            d = abs(gn - bn) / max(abs(gn), 1e-9)
            m = "exact" if d <= 0.05 else "close" if d <= 0.10 else "miss"
        else:
            ng, nb = norm(g), norm(b)
            m = "exact" if ng == nb or (ng and nb and (ng in nb or nb in ng)) or parts(g) == parts(b) or (str(g).startswith("[") and all(p in nb for p in parts(g))) else "miss"
    conf = {"exact": "high", "close": "medium", "miss": "low"}[m]
    if m == "exact" and l.get("confidence") == "low": conf = "medium"
    stats[m] += 1; stats["conf_" + conf] += 1
    rows.append({"pack_id": l["pack_id"], "id": k["id"], "gold": g, "blind": b, "unit": l.get("unit"),
                 "match": m, "labeller_conf": l.get("confidence"), "final_conf": conf,
                 "question_type": k["question_type"], "rationale": l.get("rationale")})
with open(out / "compare.jsonl", "w", encoding="utf-8", newline="\n") as f:
    for r in rows: f.write(json.dumps(r) + "\n")
n = len(rows)
print(f"items {n} | exact {stats['exact']} | close {stats['close']} | miss {stats['miss']} | key order ok {stats['key_order_ok']}/{n}")
print(f"final confidence: high {stats['conf_high']}, medium {stats['conf_medium']}, low {stats['conf_low']}")
for r in rows:
    if r["match"] != "exact":
        print(f"  {r['pack_id']} {r['match']:5} gold={r['gold']!r} blind={r['blind']!r} ({r['question_type']}) {r['rationale'][:110]}")
