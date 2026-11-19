#!/usr/bin/env python3
"""Error taxonomy v1 over committed predictions (task 3.0e, carried from 2.6). Laptop only: no GPU, no regeneration.

  python scripts/taxonomy.py results/evals/run_a_r16-adapter_test_image_only_cot_0shot_bf16
  python scripts/taxonomy.py --all --write        # every evaluation; writes taxonomy.json beside each summary

Every item gets exactly one category, assigned in this order (the first that applies wins):

  ok                     correct under the relaxed-accuracy scorer (scoring.py), and nothing else: so
                         share(ok) == relaxed_accuracy and 1 - error_rate reconciles with the reported number
  schema_invalid         the output breaks the contract; broken down by WHICH rule broke (no_json,
                         missing:<field>, bad_answer_type, evidence_not_list), not left as one bucket
  failed_to_abstain      the chart cannot answer it, and the model gave an answer
  false_abstention       the chart can answer it, and the model declined (answer null)
  answer_type_mismatch   the model's answer_type differs from the gold's (a number given for a yes/no, ...)
  scale_error            a number off by a power of ten (the 0.64-for-64% pattern; known test gold noise, §8)
  near_miss              a number within 5-15% of gold: reading between gridlines rather than the plotted value
  reasoning_error        an arithmetic/comparison number is wrong although every number cited as evidence is in
                         the chart's table: the model read the chart right and computed wrong
  extraction_error       any other wrong number: the values it read were not the chart's. Answer-only outputs
                         cite no evidence, so for them every wrong non-near, non-scale number lands here: the
                         reasoning/extraction split is only measurable for runs that show their working
  text_mismatch          a wrong text or yes/no answer

Flags are carried beside the category, never promoted to one, because they can co-occur with any of them:

  ungrounded_evidence    cites numbers, none of which are in the chart's table (on any item, including ok ones)
  refusal_reason_unstated  a correct refusal whose rationale names none of the probe's absent entities (Eval-C
                         only, where absent_entities are known): right to refuse, reason not shown
"""
import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
import check_teacher as CT  # noqa: E402
import rescore  # noqa: E402
import scoring  # noqa: E402

CATEGORIES = ("ok", "schema_invalid", "failed_to_abstain", "false_abstention", "answer_type_mismatch",
              "scale_error", "near_miss", "reasoning_error", "extraction_error", "text_mismatch")


def schema_rules(out, style):
    """The contract rules an output breaks, by name; empty when schema-valid."""
    if not isinstance(out, dict):
        return ["no_json"]
    broken = [f"missing:{f}" for f in sorted(scoring.fields_for(style) - set(out))]
    if out.get("answer_type") not in scoring.TYPES:
        broken.append("bad_answer_type")
    if style == "cot" and "evidence" in out and not isinstance(out.get("evidence"), list):
        broken.append("evidence_not_list")
    return broken


def classify(pred_row, item, style):
    """(category, schema rules broken, flags) for one committed prediction row and its frozen item."""
    res = scoring.score(item, pred_row["raw"], style)
    out = scoring.extract_json(pred_row["raw"])
    out = out if isinstance(out, dict) else None
    rules = schema_rules(out, style)
    flags = []
    if res["evidence_n"] and not res["evidence_ok"]:
        flags.append("ungrounded_evidence")
    gold_type = item["answer_type"]
    if res["correct"]:
        if gold_type == "unanswerable" and item.get("absent_entities"):
            why = " ".join(str(x) for x in [(out or {}).get("rationale")] + list((out or {}).get("evidence") or []))
            if not any(e.lower() in why.lower() for e in item["absent_entities"]):
                flags.append("refusal_reason_unstated")
        return "ok", rules, flags
    if not res["schema_valid"]:
        return "schema_invalid", rules, flags
    if gold_type == "unanswerable":
        return "failed_to_abstain", rules, flags
    if res["abstained"]:
        return "false_abstention", rules, flags
    if out.get("answer_type") != gold_type:
        return "answer_type_mismatch", rules, flags
    if gold_type == "number":
        gold = json.loads(item["messages"][2]["content"])["answer"]
        p, g = CT.num(res["pred"]), CT.num(gold)
        if p is not None and g not in (None, 0) and p != 0:
            ratio = abs(p / g)
            k = round(math.log10(ratio))
            if k != 0 and abs(ratio / 10 ** k - 1) <= 0.05:
                return "scale_error", rules, flags
            if abs(p - g) <= 0.15 * abs(g):
                return "near_miss", rules, flags
        if item["question_type"] == "arithmetic_comparison" and res["evidence_n"] and res["evidence_ok"] == res["evidence_n"]:
            return "reasoning_error", rules, flags
        return "extraction_error", rules, flags
    return "text_mismatch", rules, flags


def taxonomy(folder):
    summary = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
    s = summary["settings"]
    items = rescore.items_for(s)
    rows = [json.loads(l) for l in open(folder / "predictions.jsonl", encoding="utf-8")]
    per_item, cats, rules, flags = [], Counter(), Counter(), Counter()
    by_qtype = {}
    for r in rows:
        item = items[r["id"]]
        cat, broken, fl = classify(r, item, s["style"])
        cats[cat] += 1
        rules.update(broken if cat == "schema_invalid" else [])
        flags.update(fl)
        by_qtype.setdefault(item["question_type"], Counter())[cat] += 1
        per_item.append({"id": r["id"], "category": cat, "schema_rules": broken, "flags": fl,
                         "question_type": item["question_type"], "answer_type": item["answer_type"]})
    n = len(rows)
    ok_share = cats["ok"] / n
    if abs(ok_share - summary["overall"]["relaxed_accuracy"]) > 5e-4:
        raise SystemExit(f"{folder.name}: ok share {ok_share:.4f} does not reconcile with relaxed accuracy "
                         f"{summary['overall']['relaxed_accuracy']}")
    return {"name": folder.name, "n": n, "relaxed_accuracy": summary["overall"]["relaxed_accuracy"],
            "error_rate": round(1 - ok_share, 4),
            "categories": {c: cats[c] for c in CATEGORIES},
            "schema_rules": dict(sorted(rules.items())), "flags": dict(sorted(flags.items())),
            "by_question_type": {q: {c: v[c] for c in CATEGORIES if v[c]} for q, v in sorted(by_qtype.items())},
            "items": per_item}


def markdown(t):
    errors = t["n"] - t["categories"]["ok"]
    lines = [f"### {t['name']}", "",
             f"n = {t['n']} · relaxed accuracy {100 * t['relaxed_accuracy']:.1f}% · errors {errors} "
             f"({100 * t['error_rate']:.1f}%)", "", "| Category | Items | Share of errors |", "|---|---:|---:|"]
    for c in CATEGORIES[1:]:
        if t["categories"][c]:
            lines.append(f"| {c} | {t['categories'][c]} | {100 * t['categories'][c] / max(errors, 1):.0f}% |")
    if t["schema_rules"]:
        lines.append("\nSchema rules broken: " + ", ".join(f"{k} {v}" for k, v in t["schema_rules"].items()))
    if t["flags"]:
        lines.append("\nFlags: " + ", ".join(f"{k} {v}" for k, v in t["flags"].items()))
    lines += ["", "| Question type | " + " | ".join(CATEGORIES) + " |", "|---|" + "---:|" * len(CATEGORIES)]
    for q, v in t["by_question_type"].items():
        lines.append(f"| {q} | " + " | ".join(str(v.get(c, "")) for c in CATEGORIES) + " |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folders", nargs="*", type=Path)
    ap.add_argument("--all", action="store_true", help="every folder under results/evals")
    ap.add_argument("--write", action="store_true", help="write taxonomy.json beside each summary.json")
    args = ap.parse_args()
    folders = sorted(d for d in (REPO / "results/evals").iterdir() if (d / "summary.json").is_file()) if args.all \
        else args.folders
    if not folders:
        ap.error("name evaluation folders or pass --all")
    for folder in folders:
        t = taxonomy(folder)
        print(markdown(t) + "\n")
        if args.write:
            (folder / "taxonomy.json").write_text(json.dumps(t, indent=1, ensure_ascii=False) + "\n",
                                                  encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
