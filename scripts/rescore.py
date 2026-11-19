#!/usr/bin/env python3
"""Re-score committed evaluations with the current scripts/scoring.py, on the laptop, without a GPU (task 3.0b).

  python scripts/rescore.py            # check every results/evals/* folder, write nothing
  python scripts/rescore.py --write    # backfill new metrics into summary.json and predictions.jsonl

Every evaluation folder keeps the raw generation of every item, so a scorer change can be applied without
regenerating anything. That is only safe if the change *adds* metrics: this script re-scores each raw output
against the frozen item it was produced for, and refuses to write if any per-item verdict or any previously
reported overall number moves. A scorer change that does move a reported number is a different act (re-scoring
every baseline, see the scoring.py docstring) and must not happen through this script.

Metrics a scorer change retires (RETIRED, below) are checked against their replacement before they are dropped:
`refusal_accuracy` was true abstention reported alone, so it must equal `abstention.true_abstention` exactly.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
import scoring  # noqa: E402

PER_ITEM = ("json_valid", "schema_valid", "pred", "correct", "evidence_n", "evidence_ok")
# overall metrics replaced by a scorer change -> how to read the old value from the new summary
RETIRED = {"refusal_accuracy": lambda o: o["abstention"]["true_abstention"]}
STYLE = {"reasoning": "cot", "answer_only": "answer_only"}


def items_for(settings, root=REPO):
    sub = "answer_only/" if settings["targets"] == "answer_only" else ""
    items_dir = settings.get("items_dir", settings["data_dir"])  # eval_c and later held-out sets live outside data_dir
    path = root / items_dir / settings["variant"] / f"{sub}{settings['split']}.jsonl"
    return {r["id"]: r for r in map(json.loads, open(path, encoding="utf-8"))}


def same(a, b):
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(same(a[k], b[k]) for k in a)
    if isinstance(a, float) or isinstance(b, float):
        return a is not None and b is not None and abs(a - b) < 5e-4
    return a == b


def rescore(folder, root=REPO):
    """(new summary, new prediction rows, list of problems) for one evaluation folder."""
    summary = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
    preds = [json.loads(l) for l in open(folder / "predictions.jsonl", encoding="utf-8")]
    s = summary["settings"]
    items = items_for(s, root)
    problems, new_preds, results = [], [], []
    for p in preds:
        if p["id"] not in items:
            problems.append(f"{p['id']}: not in the frozen {s['split']} file")
            continue
        res = scoring.score(items[p["id"]], p["raw"], s["style"])
        for k in PER_ITEM:
            if not same(res[k], p[k]):
                problems.append(f"{p['id']}: {k} was {p[k]!r}, re-scores as {res[k]!r}")
        results.append(res)
        new_preds.append({**p, **{k: v for k, v in res.items() if k not in p}})
    used = [items[p["id"]] for p in preds if p["id"] in items]
    overall, by = scoring.summarise(results, used)
    old = summary["overall"]
    for k, v in old.items():
        if k in RETIRED:
            if not same(RETIRED[k](overall), v):
                problems.append(f"overall {k} {v} has no exact replacement ({RETIRED[k](overall)})")
        elif k in overall and not same(overall[k], v):
            problems.append(f"overall {k} was {v}, re-scores as {overall[k]}")
    if by != summary["by"]:
        problems.append("per-group breakdown changed")
    keep = {k: v for k, v in old.items() if k not in overall and k not in RETIRED}  # timing, truncation
    summary["overall"] = {**{k: (round(v, 4) if isinstance(v, float) else v) for k, v in overall.items()}, **keep}
    summary["overall"]["abstention"] = {k: (round(v, 4) if isinstance(v, float) else v)
                                        for k, v in overall["abstention"].items()}
    added, retired = sorted(set(overall) - set(old)), sorted(set(old) & set(RETIRED))
    if added or retired:  # a no-op re-score leaves no stamp
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True).stdout.strip()
        summary.setdefault("rescored", []).append({"by": "scripts/rescore.py", "at_commit": commit,
                                                   "added": added, "retired": retired})
    return summary, new_preds, problems


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=REPO / "results" / "evals")
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()
    failed = False
    for folder in sorted(d for d in args.root.iterdir() if (d / "summary.json").is_file()):
        summary, preds, problems = rescore(folder)
        a = summary["overall"]["abstention"]
        print(f"{folder.name:64} true {a['true_abstention']} (n={a['n_unanswerable']})  "
              f"false {a['false_abstention']} (n={a['n_answerable']})  {'OK' if not problems else 'MOVED'}")
        for prob in problems[:10]:
            print("   ", prob)
        failed |= bool(problems)
        if args.write and not problems:
            # LF on every machine: *.json is pinned to LF in .gitattributes
            (folder / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
                                                 encoding="utf-8", newline="\n")
            with open(folder / "predictions.jsonl", "w", encoding="utf-8", newline="\n") as f:
                for p in preds:
                    f.write(json.dumps(p, ensure_ascii=False) + "\n")
    if failed:
        sys.exit("a re-score moved a reported number: nothing was written for those folders")


if __name__ == "__main__":
    main()
