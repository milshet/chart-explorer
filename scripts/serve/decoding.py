"""Validation and retry (task 4.3): the two layers that enforce the answer contract at serving time.

There is no constrained decoding (no schema.json, plan §3.2), so a first attempt is not well-formed by
construction. Instead:

  validate   applies scoring.contract_errors(), the single definition of the contract, including the cross-field
             rule that an "unanswerable" answer_type carries a null answer.
  retry      re-asks with the validation errors quoted: the failed reply goes back as the assistant turn and the
             errors as a new user turn, so the model sees what it said and what was wrong with it.
  fallback   when MAX_ATTEMPTS are used up, a contract-valid abstention. The caller learns it is a fallback from
             the envelope's "degraded" flag, never from the contract object: a fallback that looked like the
             model's own refusal would land in the abstention rate the model card quotes.

Every attempt is returned, so the service can count retries and fallbacks rather than just log them: an endpoint
quietly retrying a third of its requests is a different system from the one measured at 99% schema validity.
"""
import sys
from dataclasses import dataclass, field
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1]
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
import scoring  # noqa: E402

MAX_ATTEMPTS = 3  # the first answer and up to two retries

SCHEMA_LINE = {  # the last line of the training prompt for each output style, repeated in a retry
    "cot": 'Respond with JSON: {"evidence": [string], "rationale": string, "answer_type": "number" | "text" | '
           '"yes_no" | "unanswerable", "unit": string | null, "answer": string | number | null}',
    "answer_only": 'Respond with JSON: {"answer_type": "number" | "text" | "yes_no" | "unanswerable", '
                   '"unit": string | null, "answer": string | number | null}',
}


@dataclass
class Outcome:
    result: dict
    degraded: bool
    attempts: list = field(default_factory=list)  # one dict per generation: errors, tokens, seconds


def validate(raw, style):
    """(parsed object or None, contract errors). No errors means the object is the answer."""
    out = scoring.extract_json(raw)
    if out is None:
        return None, ["the reply contains no parseable JSON object" + (" (it was cut off)" if "{" in raw else "")]
    return out, scoring.contract_errors(out, style)


def retry_prompt(errors, style):
    """The user turn that re-asks: the errors, quoted, then the schema line the model was trained on."""
    lines = "\n".join(f"- {e}" for e in errors)
    return (f"Your reply did not follow the required JSON format:\n{lines}\n"
            f"Answer the same question again. {SCHEMA_LINE[style]}")


def fallback(style):
    """A contract-valid abstention: what a caller gets when every attempt failed validation."""
    out = {"answer_type": "unanswerable", "unit": None, "answer": None}
    return {"evidence": [], "rationale": "", **out} if style == "cot" else out


def text_turn(role, text):
    return {"role": role, "content": [{"type": "text", "text": text}]}


def answer(engine, messages, images, style, max_attempts=MAX_ATTEMPTS, max_new_tokens=512):
    """Ask, validate, re-ask with the errors quoted; an Outcome either way."""
    convo, attempts = list(messages), []
    for _ in range(max_attempts):
        g = engine.generate(convo, images, max_new_tokens=max_new_tokens)
        out, errors = validate(g.raw, style)
        attempts.append({"errors": errors, "prompt_tokens": g.prompt_tokens, "new_tokens": g.new_tokens,
                         "truncated": g.truncated, "seconds": g.seconds, "raw": g.raw})
        if not errors:
            return Outcome(out, False, attempts)
        convo = convo + [text_turn("assistant", g.raw), text_turn("user", retry_prompt(errors, style))]
    return Outcome(fallback(style), True, attempts)
