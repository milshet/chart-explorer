#!/usr/bin/env python3
"""Comparison table for evaluations of one split (#8): metrics per model plus a paired test against a reference.

  python scripts/compare.py --split test --reference base_test_image_only_answer_only_2shot_4bit

Reads results/evals/*/summary.json and predictions.jsonl, keeps the reportable evaluations of --split, and
prints a Markdown table. The paired test is an exact two-sided McNemar test on the items where exactly one
of the two models is correct; every model answers the same items, so this is stronger than comparing
two accuracies with separate margins of error.
"""
import argparse
import json
import math
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
QTYPES = ("arithmetic_comparison", "retrieval", "visual_reference")  # unanswerable = the Refusals column


def mcnemar(a, b):
    """(items only a gets right, items only b gets right, exact two-sided p) for {id: correct} dicts."""
    if a.keys() != b.keys():
        raise ValueError("the two evaluations cover different items")
    x = sum(a[i] and not b[i] for i in a)
    y = sum(b[i] and not a[i] for i in a)
    n = x + y
    p = min(1.0, 2 * sum(math.comb(n, j) for j in range(min(x, y) + 1)) / 2 ** n) if n else 1.0
    return x, y, p


def load(root, split):
    evals = {}
    for summary_file in sorted(Path(root).glob("*/summary.json")):
        s = json.loads(summary_file.read_text(encoding="utf-8"))
        if s["settings"]["split"] != split or not s.get("reportable"):
            continue
        preds = [json.loads(l) for l in open(summary_file.with_name("predictions.jsonl"), encoding="utf-8")]
        evals[summary_file.parent.name] = (s, {p["id"]: p["correct"] for p in preds})
    return evals


def pct(v):
    return "" if v is None else f"{100 * v:.1f}%"


def table(evals, reference):
    ref = evals[reference][1]
    head = ["Evaluation", "Relaxed acc.", "Schema valid", "Refusals", *[q.split("_")[0].capitalize() for q in QTYPES],
            "Tokens out", "s / item", "vs reference: only this / only ref (p)"]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for name, (s, correct) in evals.items():
        o, by = s["overall"], s["by"]["question_type"]
        refusals = by.get("unanswerable", {})
        x, y, p = mcnemar(correct, ref) if name != reference else (None, None, None)
        lines.append("| " + " | ".join([
            name, pct(o["relaxed_accuracy"]), pct(o["schema_valid"]),
            f"{round(refusals['mean'] * refusals['count'])}/{refusals['count']}" if refusals else "",
            *[pct(by[q]["mean"]) if q in by else "" for q in QTYPES],
            f"{o['mean_new_tokens']}", f"{o['seconds_per_item']}",
            "reference" if p is None else f"{x} / {y} (p = {p:.3g})",
        ]) + " |")
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--split", choices=("val", "test"), required=True)
    p.add_argument("--reference", required=True, help="folder name of the evaluation to test against")
    p.add_argument("--root", default=str(REPO / "results" / "evals"))
    args = p.parse_args()
    evals = load(args.root, args.split)
    if args.reference not in evals:
        raise SystemExit(f"{args.reference} is not a reportable {args.split} evaluation in {args.root}")
    print(table(evals, args.reference))


if __name__ == "__main__":
    main()
