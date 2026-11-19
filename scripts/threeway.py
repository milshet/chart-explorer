#!/usr/bin/env python3
"""The required comparison (task 3.5) and the retrieval-failure verdict (task 3.6), from committed evaluations.

  python scripts/threeway.py            # print both tables from results/evals; NOT MEASURED where a run is missing
  python scripts/threeway.py --write    # also write results/rag/threeway.json and results/rag/threeway.md

Conditions on Eval-A (180 test items), every one scored on the SAME item ids (checked before the table is built):

  base              base model, few-shot (the Week 1 reference), image_only
  base+RAG          base model, few-shot, image_table prompt with the retrieved table (rag_test)
  fine-tuned        run_a_r16, image_only
  fine-tuned+RAG    run_a_r16 on rag_test: the prompt format is new to it (prompt_format_new_to_model)
  fine-tuned(table)+RAG  run_a_r16_table on rag_test: trained on that format (§9 decision 6)

Reading the last two together separates "the prompt format changed" from "retrieval helped". Significance is the
paired exact McNemar test (compare.mcnemar) against base and against fine-tuned, never a difference of percentages.

Task 3.6 compares, on the rows whose gold table was withheld, rag_test_withheld against the same model without
RAG. Verdict: graceful (accuracy no lower than without RAG: it fell back to the chart), harmful (lower: the wrong
table misled it). Over-abstention is a separate flag, raised when false abstention on those rows rises.
"""
import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
from compare import mcnemar  # noqa: E402

EVALS = REPO / "results" / "evals"
OUT = REPO / "results" / "rag"
CONDITIONS = {
    "base": "base_test_image_only_answer_only_2shot_4bit",
    "base+RAG": "base_rag_test_image_table_answer_only_2shot_4bit",
    "fine-tuned": "run_a_r16-adapter_test_image_only_cot_0shot_bf16",
    "fine-tuned+RAG": "run_a_r16-adapter_rag_test_image_table_cot_0shot_bf16",
    "fine-tuned(table)+RAG": "run_a_r16_table-adapter_rag_test_image_table_cot_0shot_bf16",
}
WITHHELD = {  # task 3.6: (with RAG, gold withheld for half the rows) vs (the same model without RAG)
    "fine-tuned(table)": ("run_a_r16_table-adapter_rag_test_withheld_image_table_cot_0shot_bf16",
                          "run_a_r16-adapter_test_image_only_cot_0shot_bf16"),
    "base": ("base_rag_test_withheld_image_table_answer_only_2shot_4bit",
             "base_test_image_only_answer_only_2shot_4bit"),
}


def load(folder):
    """{id: prediction row} for a committed evaluation, or None when it has not been run."""
    path = EVALS / folder / "predictions.jsonl"
    if not path.is_file():
        return None
    return {r["id"]: r for r in map(json.loads, open(path, encoding="utf-8"))}


def rates(preds, ids):
    rows = [preds[i] for i in ids]
    ans = [r for r in rows if r["answer_type"] != "unanswerable"]
    una = [r for r in rows if r["answer_type"] == "unanswerable"]
    return {"n": len(rows), "relaxed_accuracy": round(sum(r["correct"] for r in rows) / len(rows), 4),
            "true_abstention": f"{sum(r['abstained'] for r in una)}/{len(una)}",
            "false_abstention": f"{sum(r['abstained'] for r in ans)}/{len(ans)}",
            "false_abstention_rate": round(sum(r["abstained"] for r in ans) / len(ans), 4) if ans else None}


def paired(a, b, ids):
    x, y, p = mcnemar({i: a[i]["correct"] for i in ids}, {i: b[i]["correct"] for i in ids})
    return {"only_this": x, "only_other": y, "p": p}  # unrounded: 3.5e-11 must not print as 0


def comparison(conditions=CONDITIONS, loader=load):
    preds = {name: loader(folder) for name, folder in conditions.items()}
    present = {n: p for n, p in preds.items() if p is not None}
    if not present:
        return {"conditions": {n: "NOT MEASURED" for n in conditions}}
    ids = sorted(next(iter(present.values())))
    for n, p in present.items():
        if sorted(p) != ids:
            raise SystemExit(f"{n} was scored on different rows: conditions must cover identical items")
    out = {}
    for n in conditions:
        if n not in present:
            out[n] = "NOT MEASURED"
            continue
        row = rates(present[n], ids)
        for ref in ("base", "fine-tuned"):
            if ref in present and ref != n:
                row[f"vs {ref}"] = paired(present[n], present[ref], ids)
        out[n] = row
    return {"n_items": len(ids), "conditions": out}


def verdict(rag, no_rag, withheld_ids):
    """Task 3.6 on the withheld rows: graceful or harmful, plus the over-abstention flag."""
    with_rag, without = rates(rag, withheld_ids), rates(no_rag, withheld_ids)
    test = paired(rag, no_rag, withheld_ids)
    harmful = with_rag["relaxed_accuracy"] < without["relaxed_accuracy"]
    over = (with_rag["false_abstention_rate"] or 0) > (without["false_abstention_rate"] or 0)
    return {"n_withheld": len(withheld_ids), "with_rag": with_rag, "without_rag": without, "paired": test,
            "verdict": "harmful" if harmful else "graceful", "over_abstention": over}


def degradation(withheld=WITHHELD, loader=load):
    rows = [json.loads(l) for l in open(REPO / "data/rag/image_table/rag_test_withheld.jsonl", encoding="utf-8")]
    ids = sorted(r["id"] for r in rows if r["withheld"])
    out = {}
    for name, (rag_folder, base_folder) in withheld.items():
        rag, base = loader(rag_folder), loader(base_folder)
        out[name] = "NOT MEASURED" if rag is None or base is None else verdict(rag, base, ids)
    return out


def markdown(cmp, deg):
    lines = ["| Condition | Relaxed acc. | True abst. | False abst. | vs base (only this / only base, p) | vs fine-tuned |",
             "|---|---|---|---|---|---|"]
    for n, r in cmp["conditions"].items():
        if r == "NOT MEASURED":
            lines.append(f"| {n} | NOT MEASURED | | | | |")
            continue
        def fmt(k):
            v = r.get(k)
            return f"{v['only_this']} / {v['only_other']} (p = {v['p']:.3g})" if v else ""
        lines.append(f"| {n} | {100 * r['relaxed_accuracy']:.1f}% | {r['true_abstention']} | {r['false_abstention']} | "
                     f"{fmt('vs base')} | {fmt('vs fine-tuned')} |")
    lines += ["", "Retrieval failure (task 3.6), rows whose gold table was withheld:", ""]
    for n, v in deg.items():
        if v == "NOT MEASURED":
            lines.append(f"- {n}: NOT MEASURED")
        else:
            lines.append(f"- {n}: **{v['verdict']}**{' + over-abstaining' if v['over_abstention'] else ''} — "
                         f"{100 * v['with_rag']['relaxed_accuracy']:.1f}% with RAG vs "
                         f"{100 * v['without_rag']['relaxed_accuracy']:.1f}% without (n = {v['n_withheld']}, "
                         f"p = {v['paired']['p']:.3g}); false abstention {v['with_rag']['false_abstention']} vs "
                         f"{v['without_rag']['false_abstention']}")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()
    cmp, deg = comparison(), degradation()
    md = markdown(cmp, deg)
    print(md)
    if args.write:
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / "threeway.json").write_text(json.dumps({"comparison": cmp, "degradation": deg}, indent=1) + "\n",
                                           encoding="utf-8", newline="\n")
        (OUT / "threeway.md").write_text(md + "\n", encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
