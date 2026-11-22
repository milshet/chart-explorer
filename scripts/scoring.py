"""The Week 1 ChartQA scorer: one definition of "correct" for the notebook, evaluate.py and the write-up.

Moved from notebooks/01_vlm_baselines.ipynb (cell 6) with one fix: an escaped quote (\\") inside a JSON
string used to end the string, so such outputs counted as invalid JSON. Changing anything here changes
every reported number, so a change needs the baselines re-scored (#19, #8).

  relaxed accuracy  numbers within 5% of gold; text and yes/no equal after normalising case and
                    punctuation; unanswerable items correct when the model returns "answer": null
  json_valid        the output contains a parseable JSON object
  schema_valid      it has the fields of its target style and an allowed answer_type
  evidence grounded share of numbers cited as evidence that appear in the chart's data table
  abstention        a pair, never one number alone (task 3.0b):
                      true  = of the items the chart cannot answer, the share the model declined
                      false = of the items the chart can answer, the share the model declined anyway
                    "Declined" means a parseable object whose answer is null: the contract's abstain path.
                    Unparseable output is not a refusal, it is a format failure, and counts in neither.
                    Without the false half a wrong number and a refusal both score zero, so a model that
                    refuses everything it finds hard is indistinguishable from one that guesses.
"""
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import check_teacher as CT  # noqa: E402  norm() and num(), shared with the teacher-label checks

TYPES = {"number", "text", "yes_no", "unanswerable"}
STYLES = ("answer_only", "cot")  # cot = "reasoning" targets: evidence and rationale before the answer
GROUPS = ("question_type", "answer_type", "source", "chart_type")


def extract_json(text):
    """The first balanced {...} object in text (code fences allowed), or None."""
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
    start = text.find("{")
    if start < 0:
        return None
    depth, in_str, esc = 0, False, False
    for i, ch in enumerate(text[start:], start):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start:i + 1])
                except json.JSONDecodeError:
                    return None
    return None


def fields_for(style):
    return {"answer", "answer_type", "unit"} | ({"evidence", "rationale"} if style == "cot" else set())


def relaxed_match(pred, gold, answer_type):
    if answer_type == "unanswerable":
        return pred is None or (isinstance(pred, str) and pred.strip().lower() in ("", "null", "none"))
    if pred is None:
        return False
    if answer_type == "number":
        p, g = CT.num(pred), CT.num(gold)
        if p is None or g is None:
            return False
        return abs(p - g) <= 0.05 * abs(g) if g != 0 else abs(p) < 1e-9
    return CT.norm(pred) == CT.norm(gold)


def nums(s):
    return [float(x.replace(",", "")) for x in re.findall(r"(?<![A-Za-z\d.])\d[\d,]*(?:\.\d+)?", str(s))]


def score(item, raw, style):
    """Score one raw model output against the item's gold assistant turn."""
    if style not in STYLES:
        raise ValueError(f"style must be one of {STYLES}, not {style!r}")
    out = extract_json(raw)
    if not isinstance(out, dict):
        out = None
    gold = json.loads(item["messages"][2]["content"])
    res = {"id": item["id"], "raw": raw, "json_valid": out is not None}
    res["schema_valid"] = bool(out) and set(out) >= fields_for(style) and out.get("answer_type") in TYPES \
        and (style != "cot" or isinstance(out.get("evidence"), list))
    res["pred"] = out.get("answer") if out else None
    res["correct"] = bool(out) and relaxed_match(res["pred"], gold["answer"], item["answer_type"])
    res["abstained"] = bool(out) and relaxed_match(res["pred"], None, "unanswerable")
    ev = [e for e in (out or {}).get("evidence") or [] if isinstance(e, str)]
    # Evidence grounding: numbers the model cites that really are in the chart's data table
    # (the model never sees that table in image_only).
    cited = [v for e in ev for v in nums(e)]
    table = nums(item["input_text"])
    res["evidence_n"] = len(cited)
    res["evidence_ok"] = sum(any(abs(v - t) <= max(0.051, abs(t) * 0.01) for t in table) for v in cited)
    return res


def contract_errors(out, style):
    """Every way a parsed output breaks the answer contract; [] means it is a valid answer object.

    The serving check (task 4.3): stricter than schema_valid, which score() keeps unchanged so no reported number
    moves. On top of schema_valid it requires exactly the contract's fields, string evidence and rationale, a
    string-or-null unit and rationale, an answer that is a string, a number or null, and the cross-field rule both ways: the
    answer is null if and only if answer_type is "unanswerable". Every gold answer in the dataset satisfies it.
    Messages are written to be quoted back to the model in a retry.
    """
    if not isinstance(out, dict):
        return ["the reply is not a JSON object"]
    errors, need = [], fields_for(style)
    missing, extra = sorted(need - set(out)), sorted(set(out) - need)
    if missing:
        errors.append(f"missing field(s): {', '.join(missing)}")
    if extra:
        errors.append(f"unexpected field(s): {', '.join(extra)}")
    kind = out.get("answer_type")
    if "answer_type" in out and kind not in TYPES:
        errors.append(f'"answer_type" must be one of {", ".join(sorted(TYPES))}, not {kind!r}')
    if "unit" in out and out["unit"] is not None and not isinstance(out["unit"], str):
        errors.append('"unit" must be a string or null')
    answer = out.get("answer")
    if "answer" in out and (isinstance(answer, bool) or not isinstance(answer, (str, int, float, type(None)))):
        errors.append('"answer" must be a string, a number or null')
    if "answer" in out and kind in TYPES:
        if kind == "unanswerable" and answer is not None:
            errors.append('"answer_type" is "unanswerable", so "answer" must be null')
        if kind != "unanswerable" and answer is None:
            errors.append(f'"answer" is null, so "answer_type" must be "unanswerable", not "{kind}"')
    if style == "cot":
        ev = out.get("evidence")
        if "evidence" in out and not (isinstance(ev, list) and all(isinstance(e, str) for e in ev)):
            errors.append('"evidence" must be a list of strings')
        if "rationale" in out and not isinstance(out["rationale"], (str, type(None))):
            errors.append('"rationale" must be a string')  # null allowed: Eval-C's gold probes carry it
    return errors


def _mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else None


def abstention(rows):
    """The abstention pair over scored rows that carry the item's gold answer_type.

    Returned as one object so neither rate can be reported without the other; a rate is None when its
    denominator is empty, and the denominators travel with the rates so a 9/9 reads as nine items.
    """
    una = [r for r in rows if r["answer_type"] == "unanswerable"]
    ans = [r for r in rows if r["answer_type"] != "unanswerable"]
    return {"true_abstention": _mean(r["abstained"] for r in una), "n_unanswerable": len(una),
            "false_abstention": _mean(r["abstained"] for r in ans), "n_answerable": len(ans)}


def summarise(results, items):
    """Overall metrics and accuracy per group; the same numbers as the notebook's pandas summary."""
    by_id = {it["id"]: it for it in items}
    rows = [{**r, **{g: by_id[r["id"]][g] for g in GROUPS}} for r in results]
    overall = {
        "n": len(rows),
        "relaxed_accuracy": _mean(r["correct"] for r in rows),
        "json_valid": _mean(r["json_valid"] for r in rows),
        "schema_valid": _mean(r["schema_valid"] for r in rows),
        "evidence_grounded": sum(r["evidence_ok"] for r in rows) / max(sum(r["evidence_n"] for r in rows), 1),
        "abstention": abstention(rows),
        "answerable_accuracy": _mean(r["correct"] for r in rows if r["answer_type"] != "unanswerable"),
    }
    by = {}
    for g in GROUPS:
        groups = {}
        for r in rows:
            groups.setdefault(r[g], []).append(r["correct"])
        by[g] = {k: {"mean": round(_mean(v), 3), "count": len(v)} for k, v in sorted(groups.items())}
    return overall, by
