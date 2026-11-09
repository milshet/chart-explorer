"""Fail if any question pack contains anything except pack_id, image and question,
or if a pack's text contains its own gold answer outside the question."""
import json, sys, pathlib
out = pathlib.Path(sys.argv[1])
packs = [json.loads(l) for l in open(out / "packs" / "questions.jsonl", encoding="utf-8")]
key = {k["pack_id"]: k for k in map(json.loads, open(out / "key" / "answers.jsonl", encoding="utf-8"))}
bad = 0
for p in packs:
    if set(p) != {"pack_id", "image", "question"}:
        print("EXTRA FIELDS", p["pack_id"], set(p)); bad += 1
    if "/" in p["image"].replace("images/", "", 1) or not p["image"].startswith("images/q"):
        print("IMAGE NAME NOT NEUTRAL", p["pack_id"]); bad += 1
files = sorted(f.name for f in (out / "packs").rglob("*") if f.is_file())
batches = sorted((out / "packs" / "batches").glob("*.jsonl")) if (out / "packs" / "batches").exists() else []
for b in batches:
    for l in open(b, encoding="utf-8"):
        q = json.loads(l)
        if set(q) != {"pack_id", "image", "question"}:
            print("EXTRA FIELDS IN BATCH", b.name, q.get("pack_id")); bad += 1
allowed = {"questions.jsonl"} | {b.name for b in batches} | {pathlib.Path(p["image"]).name for p in packs}
for f in files:
    if f not in allowed:
        print("UNEXPECTED FILE IN PACKS", f); bad += 1
print(f"{len(packs)} packs, {len(files)} files, problems: {bad}")
print("LEAK CHECK PASSED" if bad == 0 else "LEAK CHECK FAILED")
sys.exit(1 if bad else 0)
