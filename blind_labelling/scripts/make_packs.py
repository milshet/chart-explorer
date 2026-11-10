"""Build answer-free question packs for blind labelling.

Packs hold ONLY: pack id, a renamed copy of the chart image, the question.
The real answers go to a separate key file that the labeller is never given.
"""
import argparse, json, random, shutil, pathlib

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True)          # e.g. data/v1.0/image_only/train.jsonl
ap.add_argument("--root", required=True)          # data/v1.0 (image paths are relative to it)
ap.add_argument("--out", required=True)           # output folder for this run
ap.add_argument("--n", type=int, default=0)       # 0 = all items
ap.add_argument("--domains", default="finance,economy")
ap.add_argument("--seed", type=int, default=7)
ap.add_argument("--batch", type=int, default=0)   # also split questions into batches of this size
a = ap.parse_args()

rows = [json.loads(l) for l in open(a.data, encoding="utf-8")]
doms = set(a.domains.split(",")) if a.domains else None
if doms:
    rows = [r for r in rows if doms & set(r["domain_tags"]) or r["answer_type"] == "unanswerable"]  # refusals carry no domain tag
if a.n:
    rnd = random.Random(a.seed)
    quota = {"number": round(a.n * .6), "text": round(a.n * .2), "yes_no": round(a.n * .1)}
    quota["unanswerable"] = a.n - sum(quota.values())
    pick = []
    for t, k in quota.items():
        pool = [r for r in rows if r["answer_type"] == t]
        pick += rnd.sample(pool, min(k, len(pool)))
    rnd.shuffle(pick)
    rows = pick

out = pathlib.Path(a.out)
(out / "packs" / "images").mkdir(parents=True, exist_ok=True)
(out / "key").mkdir(parents=True, exist_ok=True)
with open(out / "packs" / "questions.jsonl", "w", encoding="utf-8", newline="\n") as fp, open(out / "key" / "answers.jsonl", "w", encoding="utf-8", newline="\n") as fk:
    for i, r in enumerate(rows, 1):
        pid = f"q{i:04d}"
        img = f"images/{pid}.png"
        shutil.copy(pathlib.Path(a.root) / r["image"], out / "packs" / img)
        fp.write(json.dumps({"pack_id": pid, "image": img, "question": r["question"]}) + "\n")
        fk.write(json.dumps({"pack_id": pid, "id": r["id"], "answer": r["answer"],
                             "answer_type": r["answer_type"], "gold_answer_raw": r["gold_answer_raw"],
                             "domain_tags": r["domain_tags"], "question_type": r["question_type"],
                             "image_readable": r["flags"].get("image_readable")}) + "\n")
if a.batch:
    qs = open(out / "packs" / "questions.jsonl", encoding="utf-8").read().splitlines()
    (out / "packs" / "batches").mkdir(exist_ok=True)
    for j in range(0, len(qs), a.batch):
        (out / "packs" / "batches" / f"b{j // a.batch:02d}.jsonl").write_bytes(("\n".join(qs[j:j + a.batch]) + "\n").encode("utf-8"))
print(f"wrote {len(rows)} packs to {out}")
