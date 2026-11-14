"""The Week 1 ChartQA scorer: one definition of "correct" for the notebook, evaluate.py and the write-up.

Moved from notebooks/01_vlm_baselines.ipynb (cell 6) with one fix: an escaped quote (\\") inside a JSON
string used to end the string, so such outputs counted as invalid JSON. Changing anything here changes
every reported number, so a change needs the baselines re-scored (#19, #8).

  relaxed accuracy  numbers within 5% of gold; text and yes/no equal after normalising case and
                    punctuation; unanswerable items correct when the model returns "answer": null
  json_valid        the output contains a parseable JSON object
  schema_valid      it has the fields of its target style and an allowed answer_type
  evidence grounded share of numbers cited as evidence that appear in the chart's data table
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
    ev = [e for e in (out or {}).get("evidence") or [] if isinstance(e, str)]
    # Evidence grounding: numbers the model cites that really are in the chart's data table
    # (the model never sees that table in image_only).
    cited = [v for e in ev for v in nums(e)]
    table = nums(item["input_text"])
    res["evidence_n"] = len(cited)
    res["evidence_ok"] = sum(any(abs(v - t) <= max(0.051, abs(t) * 0.01) for t in table) for v in cited)
    return res


def _mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else None


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
        "refusal_accuracy": _mean(r["correct"] for r in rows if r["answer_type"] == "unanswerable"),
        "answerable_accuracy": _mean(r["correct"] for r in rows if r["answer_type"] != "unanswerable"),
    }
    by = {}
    for g in GROUPS:
        groups = {}
        for r in rows:
            groups.setdefault(r[g], []).append(r["correct"])
        by[g] = {k: {"mean": round(_mean(v), 3), "count": len(v)} for k, v in sorted(groups.items())}
    return overall, by
