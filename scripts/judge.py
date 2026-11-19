#!/usr/bin/env python3
"""LLM-as-judge rubric (task 3.7): correctness, grounding and appropriate abstention, graded independently.

  python scripts/judge.py sample --name run_a_r16             # draw the judged sample; writes results/judge/<name>/sample.jsonl
  python scripts/judge.py run --name run_a_r16                # call the judge (needs Anthropic credentials; costs money)
  python scripts/judge.py spotcheck --name run_a_r16          # write judge_spot_check.jsonl for two human graders
  python scripts/judge.py agree --name run_a_r16              # judge vs humans: raw agreement AND Cohen's kappa

Three axes, graded separately because they come apart: an answer can be right with invented evidence, or well
grounded and wrong. Correctness and grounding are yes / no / n/a; abstention is graded on every item as yes / no
("was declining, or answering, the right call here").

The sample mixes Eval-A AND Eval-C: the abstention axis grades both directions, and Eval-A holds only nine
unanswerable items, so a sample from it alone would grade half the axis while the score reads as a whole one.

A verdict the parser cannot read, or a refusal from the judge, is EXCLUDED and counted (n_excluded), never scored
as zero: otherwise a judge whose output format drifted reads as a system that got worse. Grounding has a
programmatic cross-check this project already owns, the scorer's evidence_grounded; disagreements are listed.

Human agreement is reported as Cohen's kappa beside raw agreement: these axes are skewed, and 0.9 raw agreement
on a 90/10 split is what always-say-yes scores.

The judge reads text only (question, gold answer, the chart's data table, the model's output); it never sees the
image, so it grades against the table, which is the ground truth the labels came from.
"""
import argparse
import collections
import json
import random
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
import rescore  # noqa: E402  items_for

OUT = REPO / "results" / "judge"
AXES = ("correctness", "grounding", "abstention")
VERDICTS = ("yes", "no", "n/a")
SAMPLE = {"test": 60, "eval_c": 20}   # cut rule: the judge sample size is the second thing cut if the week runs short
SEED = 61003
MODEL = "claude-opus-5-5"
RUBRIC = """You are grading one answer from a chart question-answering system used in regulated financial reporting.
You see the question, the gold answer, the chart's underlying data table, and the system's raw output (JSON with
evidence, rationale, answer_type, unit, answer). Grade three axes independently:

- correctness: is the system's answer the gold answer (numbers within 5%)? "n/a" when the gold says the question
  is unanswerable.
- grounding: are the values the system cites as evidence actually in the data table (or correct computations from
  it)? "n/a" when it cites no evidence.
- abstention: was answering, or declining (answer null), the right call? The question is unanswerable exactly when
  the gold answer is null. Always "yes" or "no".

Reply with only a JSON object: {"correctness": "yes"|"no"|"n/a", "grounding": "yes"|"no"|"n/a",
"abstention": "yes"|"no", "note": "<one sentence>"}"""


def eval_folder(name, split):
    matches = sorted(p.parent for p in (REPO / "results/evals").glob(f"{name}*_{split}_*/summary.json")
                     if "_limit" not in p.parent.name)
    return matches[0] if matches else None


def draw(name):
    """The judged sample: rows from Eval-A test and Eval-C for one model's committed predictions."""
    rows, missing = [], []
    for split, n in SAMPLE.items():
        folder = eval_folder(name, split)
        if folder is None:
            missing.append(split)
            continue
        settings = json.loads((folder / "summary.json").read_text(encoding="utf-8"))["settings"]
        items = rescore.items_for(settings)
        preds = [json.loads(l) for l in open(folder / "predictions.jsonl", encoding="utf-8")]
        rng = random.Random(f"{SEED}-{split}")
        una = [p for p in preds if p["answer_type"] == "unanswerable"]
        ans = [p for p in preds if p["answer_type"] != "unanswerable"]
        # every unanswerable test item (only nine exist), the rest answerable; Eval-C is all unanswerable
        picked = una + rng.sample(ans, min(len(ans), n - len(una))) if split == "test" else rng.sample(preds, min(n, len(preds)))
        for p in sorted(picked, key=lambda p: p["id"]):
            it = items[p["id"]]
            gold = json.loads(it["messages"][2]["content"])["answer"]
            rows.append({"id": p["id"], "split": split, "eval": folder.name, "question": it["question"], "gold": gold,
                         "table": it["input_text"], "raw": p["raw"], "scorer_correct": p["correct"],
                         "scorer_evidence": [p["evidence_ok"], p["evidence_n"]]})
    return rows, missing


def prompt(row):
    return (f"Question: {row['question']}\nGold answer: {json.dumps(row['gold'])}\n\nData table:\n{row['table']}\n\n"
            f"System output:\n{row['raw']}")


def parse(text):
    """A verdict dict, or None when the reply is not a readable verdict (excluded, never scored zero)."""
    sys.path.insert(0, str(REPO / "scripts"))
    import scoring
    v = scoring.extract_json(text or "")
    if not isinstance(v, dict):
        return None
    out = {a: str(v.get(a, "")).strip().lower() for a in AXES}
    if any(out[a] not in VERDICTS for a in AXES) or out["abstention"] == "n/a":
        return None
    return {**out, "note": v.get("note")}


def call_judge(client, row, model):
    response = client.beta.messages.create(
        model=model, max_tokens=16000, system=RUBRIC,
        output_config={"effort": "medium"},                # set explicitly: the default differs by model
        betas=["server-side-fallback-2026-07-01"], fallbacks="default",
        messages=[{"role": "user", "content": prompt(row)}])
    if response.stop_reason == "refusal":
        return None, "refusal"
    text = "".join(b.text for b in response.content if b.type == "text")
    return text, response._request_id


def summarise(judged):
    ok = [j for j in judged if j["verdict"]]
    out = {"n": len(judged), "n_excluded": len(judged) - len(ok)}
    for a in AXES:
        graded = [j["verdict"][a] for j in ok if j["verdict"][a] != "n/a"]
        out[a] = {"yes": sum(v == "yes" for v in graded), "n": len(graded)}
    # the scorer's grounding vs the judge's, where both have an opinion
    out["grounding_disagreements"] = [j["id"] for j in ok if j["verdict"]["grounding"] != "n/a" and j["scorer_evidence"][1]
                                      and (j["verdict"]["grounding"] == "yes") != (j["scorer_evidence"][0] == j["scorer_evidence"][1])]
    return out


def cohen(a, b):
    """(n, raw agreement, Cohen's kappa) over items both raters graded."""
    pairs = [(x, y) for x, y in zip(a, b) if x and y]
    n = len(pairs)
    if not n:
        return 0, None, None
    po = sum(x == y for x, y in pairs) / n
    ca, cb = collections.Counter(x for x, _ in pairs), collections.Counter(y for _, y in pairs)
    pe = sum(ca[k] * cb[k] for k in set(ca) | set(cb)) / n / n
    return n, round(po, 3), (round((po - pe) / (1 - pe), 3) if pe < 1 else None)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=("sample", "run", "spotcheck", "agree"))
    ap.add_argument("--name", required=True, help="evaluation folder prefix, e.g. run_a_r16")
    ap.add_argument("--model", default=MODEL)
    args = ap.parse_args()
    d = OUT / args.name
    d.mkdir(parents=True, exist_ok=True)
    if args.step == "sample":
        rows, missing = draw(args.name)
        (d / "sample.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8", newline="\n")
        print(f"{len(rows)} rows; not yet evaluated, so not sampled: {missing or 'none'}")
    elif args.step == "run":
        import anthropic
        client = anthropic.Anthropic()
        rows = [json.loads(l) for l in open(d / "sample.jsonl", encoding="utf-8")]
        judged = []
        for r in rows:
            text, req = call_judge(client, r, args.model)
            judged.append({**r, "judge_model": args.model, "judge_raw": text, "request_id": req, "verdict": parse(text)})
        (d / "judged.jsonl").write_text("".join(json.dumps(j, ensure_ascii=False) + "\n" for j in judged), encoding="utf-8", newline="\n")
        s = summarise(judged)
        (d / "summary.json").write_text(json.dumps(s, indent=1) + "\n", encoding="utf-8", newline="\n")
        print(json.dumps(s, indent=1))
    elif args.step == "spotcheck":
        judged = [json.loads(l) for l in open(d / "judged.jsonl", encoding="utf-8")]
        pick = random.Random(SEED).sample(judged, min(30, len(judged)))
        blank = {"grader_1": {a: None for a in AXES}, "grader_2": {a: None for a in AXES}}
        (d / "judge_spot_check.jsonl").write_text("".join(json.dumps({k: j[k] for k in ("id", "question", "gold", "table", "raw")} | blank,
                                                                     ensure_ascii=False) + "\n" for j in pick), encoding="utf-8", newline="\n")
        print(f"wrote {len(pick)} items; two people fill grader_1 and grader_2 independently")
    else:
        judged = {j["id"]: j for j in map(json.loads, open(d / "judged.jsonl", encoding="utf-8"))}
        spot = [json.loads(l) for l in open(d / "judge_spot_check.jsonl", encoding="utf-8")]
        report = {}
        for a in AXES:
            j = [(judged[s["id"]]["verdict"] or {}).get(a) for s in spot]
            g1, g2 = [s["grader_1"][a] for s in spot], [s["grader_2"][a] for s in spot]
            report[a] = {"human_vs_human": cohen(g1, g2), "judge_vs_grader_1": cohen(j, g1), "judge_vs_grader_2": cohen(j, g2)}
        (d / "agreement.json").write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8", newline="\n")
        print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
