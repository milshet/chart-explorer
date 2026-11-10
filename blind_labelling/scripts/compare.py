"""Compare blind labels with the real answers (run only AFTER labelling).

Strict exact match (v2.0, decided 2026-11-10: only 100% matches are kept).
  numbers: the same value (commas and a % sign ignored; 46 and 0.46 are NOT the same). When both
           have decimals, they only need to agree to the shorter one's decimals, since that is the
           same answer rounded: 3.22 = 3.22222, 33.3 = 33.33, but 7.5 != 7.4 and 6.4 != 6
  text:    the same after lower-casing and removing punctuation, ignoring list order
           ("A and B" = "[B, A]") and a qualifier in brackets ("Naomi Osaka (tennis)" = "Naomi Osaka")
  unanswerable: both answers null

Final confidence:
  high   = exact match
  medium = exact match, but the labeller said its own confidence was low
  low    = anything else -> left out of training
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
def decimals(x):
    t = str(x).replace(",", "").replace("%", "").strip()
    return len(t.split(".")[1]) if "." in t else 0
def same_number(g, b, gn, bn):
    if abs(gn - bn) <= 1e-9 * max(1.0, abs(gn)): return True
    d = min(decimals(g), decimals(b))
    return d > 0 and abs(gn - bn) <= 0.5 * 10 ** -d + 1e-9
def norm(x):
    s = re.sub(r"\([^)]*\)", " ", str(x).lower())          # drop a qualifier in brackets
    s = re.sub(r"['\u2019]", "", s).replace("grey", "gray")
    s = re.sub(r"\.(?=\s|$)", "", s)                       # "U.S." = "u.s"
    return re.sub(r"[^a-z0-9.]+", " ", s).strip()
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
            m = "exact" if same_number(g, b, gn, bn) else "miss"
        else:
            m = "exact" if norm(g) == norm(b) or parts(g) == parts(b) else "miss"
    conf = {"exact": "high", "miss": "low"}[m]
    if m == "exact" and l.get("confidence") == "low": conf = "medium"
    stats[m] += 1; stats["conf_" + conf] += 1
    rows.append({"pack_id": l["pack_id"], "id": k["id"], "gold": g, "blind": b, "unit": l.get("unit"),
                 "match": m, "labeller_conf": l.get("confidence"), "final_conf": conf,
                 "question_type": k["question_type"], "rationale": l.get("rationale")})
with open(out / "compare.jsonl", "w", encoding="utf-8", newline="\n") as f:
    for r in rows: f.write(json.dumps(r) + "\n")
n = len(rows)
print(f"items {n} | exact {stats['exact']} | miss {stats['miss']} | key order ok {stats['key_order_ok']}/{n}")
print(f"final confidence: high {stats['conf_high']}, medium {stats['conf_medium']}, low {stats['conf_low']}")
for r in rows:
    if r["match"] != "exact":
        print(f"  {r['pack_id']} {r['match']:5} gold={r['gold']!r} blind={r['blind']!r} ({r['question_type']}) {r['rationale'][:110]}")
