"""Scoring for ChartAudit baselines and fine-tuned runs.

Use this one module, unchanged, for every run that is compared, so before and after
numbers come from the same rules. Bump SCORER_VERSION when a rule changes and re-score
saved predictions rather than correcting individual results by hand.
"""
import json
import re

import check_teacher as CT

SCORER_VERSION = "1.1"

FIELDS = {"answer", "answer_type", "unit", "evidence", "rationale"}
TYPES = {"number", "text", "yes_no", "unanswerable"}

YEAR_QUESTION = re.compile(
    r"\b(?:which|what)\s+(?:\d+\s+|two\s+)?years?\b"
    r"|^\s*when\b"
    r"|\bwhen\s+(?:did|does|do|was|is|were)\b",
    re.IGNORECASE,
)
ITEM_SEPARATOR = re.compile(r"\s*(?:,|;|&|\band\b)\s*", re.IGNORECASE)


def extract_json(text):
    """Return the first JSON object in text, or None if there is no complete one."""
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


def is_list_gold(gold):
    """Multi-part golds are stored as literal strings, such as '[Slovenia, Turkey]'."""
    if not isinstance(gold, str):
        return False
    gold = gold.strip()
    return gold.startswith("[") and gold.endswith("]")


def split_items(value):
    """Split a list-like answer on commas, semicolons, 'and' and '&'."""
    if isinstance(value, (list, tuple)):
        return [str(v).strip() for v in value if str(v).strip()]
    text = str(value).strip()
    if text.startswith("[") and text.endswith("]"):
        text = text[1:-1]
    return [part.strip() for part in ITEM_SEPARATOR.split(text) if part.strip()]


def is_year_value(value):
    number = CT.num(value)
    return number is not None and float(number).is_integer() and 1800 <= number <= 2100


def is_year_question(question, gold):
    """True when the question asks for a year and every gold item looks like one."""
    if not question or not YEAR_QUESTION.search(question):
        return False
    golds = split_items(gold) if is_list_gold(gold) else [gold]
    return bool(golds) and all(is_year_value(g) for g in golds)


def _item_matches(pred, gold, exact_numbers):
    p, g = CT.num(pred), CT.num(gold)
    if p is not None and g is not None:
        if exact_numbers:
            return p == g
        return abs(p - g) <= 0.05 * abs(g) if g != 0 else abs(p) < 1e-9
    return CT.norm(pred) == CT.norm(gold)


def _list_matches(pred, gold, exact_numbers):
    """Same items in any order; numeric items use the numeric rule."""
    preds, golds = split_items(pred), split_items(gold)
    if len(preds) != len(golds):
        return False
    remaining = list(preds)
    for g in golds:
        for p in remaining:
            if _item_matches(p, g, exact_numbers):
                remaining.remove(p)
                break
        else:
            return False
    return True


def relaxed_match(pred, gold, answer_type, question=None):
    """Numbers within 5% of gold, except years, which must match exactly."""
    if answer_type == "unanswerable":
        return pred is None or (isinstance(pred, str) and pred.strip().lower() in ("", "null", "none"))
    if pred is None:
        return False
    exact_numbers = is_year_question(question, gold)
    if is_list_gold(gold):
        return _list_matches(pred, gold, exact_numbers)
    if answer_type == "number":
        if CT.num(pred) is None or CT.num(gold) is None:
            return False
        return _item_matches(pred, gold, exact_numbers)
    return CT.norm(pred) == CT.norm(gold)


def word_count(value):
    """Words in an answer. A list-style gold counts one per item."""
    if value is None:
        return 0
    if isinstance(value, (list, tuple)):
        return sum(len(str(v).split()) for v in value)
    if is_list_gold(value):
        return sum(len(item.split()) for item in split_items(value))
    return len(str(value).split())


def schema_valid(out):
    return (
        bool(out)
        and set(out) >= FIELDS
        and isinstance(out.get("answer_type"), str)
        and out["answer_type"] in TYPES
        and isinstance(out.get("evidence"), list)
    )


def score(item, raw, truncated=False):
    """Score one raw model output against its item."""
    out = extract_json(raw)
    if not isinstance(out, dict):
        out = None
    gold = json.loads(item["messages"][2]["content"])
    pred = out.get("answer") if out else None
    # A missing answer key is not a refusal, so it must not score as one.
    has_answer = bool(out) and "answer" in out
    evidence = [e for e in (out or {}).get("evidence") or [] if isinstance(e, str)] if out else []
    return {
        "id": item["id"],
        "raw": raw,
        "truncated": bool(truncated),
        "json_valid": out is not None,
        "schema_valid": schema_valid(out),
        "pred": pred,
        "correct": has_answer and relaxed_match(pred, gold["answer"], item["answer_type"], item.get("question")),
        "pred_words": word_count(pred),
        "gold_words": word_count(gold["answer"]),
        "evidence_n": len(evidence),
        "evidence_ok": sum(CT.check_evidence(e, item["input_text"]) is None for e in evidence),
    }
