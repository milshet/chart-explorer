"""The single generation path behind every Week 4 number (task 4.1; 4.2 and 4.3 reuse it).

A merge check, a quantisation delta and a served latency are only comparable if the two sides answered through
the same code with the same token budget. So there is one engine here, and the merge verification (4.1), the
quantisation delta (4.2) and the endpoint (4.3) all call it:

  HFEngine        transformers.generate, bf16, batch 1, greedy (decision 12: no vLLM). Loads either a merged
                  checkpoint directory, or the base model plus an adapter (the unmerged reference 4.1 compares
                  against). Refuses a quantised checkpoint unless allow_quantized (task 4.2 scores one on purpose).
  answer_items    runs items through an engine and scores each answer with scripts/scoring.py.
  parsed_answer   the answer an output commits to, normalised: what "the same answer" means when two models are
  same_answer     compared. Parsed, not raw: two outputs that differ in whitespace or rationale wording but commit
                  to the same answer agree; a check that fires on whitespace gets ignored the second time.

Prompts are built by evaluate.build(), exactly as in training and in every reported evaluation. The token budget
is evaluate.py's default (512), so a served answer has the same room as a scored one. Nothing here imports torch
at module level: the comparison logic is tested on the laptop.
"""
import math
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1]
REPO = SCRIPTS.parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
import check_teacher as CT  # noqa: E402  norm() and num(), the scorer's own normalisers
import evaluate as ev  # noqa: E402  build(), prompt_text(), stop_ids(), answer_length()
import scoring  # noqa: E402

MAX_NEW_TOKENS = 512  # evaluate.py's default: every reported number was generated with this budget


@dataclass
class Generation:
    raw: str
    prompt_tokens: int
    new_tokens: int
    truncated: bool
    seconds: float


class HFEngine:
    """bf16 transformers.generate, batch 1, greedy: a merged checkpoint, or base + adapter (never both)."""

    def __init__(self, model_dir=None, adapter=None, device=0, allow_quantized=False):
        if model_dir and adapter:
            raise ValueError("give a merged checkpoint or an adapter, not both")
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor
        self.torch = torch
        src = str(model_dir) if model_dir else ev.st.MODEL_ID
        self.processor = AutoProcessor.from_pretrained(src, local_files_only=True)
        model = AutoModelForImageTextToText.from_pretrained(src, dtype=torch.bfloat16, device_map={"": device},
                                                            local_files_only=True)
        quant = getattr(model.config, "quantization_config", None)
        if quant and not allow_quantized:
            raise RuntimeError(f"{src} is a quantised checkpoint; pass allow_quantized=True to score one on purpose")
        if adapter:
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, str(adapter), local_files_only=True)
        self.model = model.eval()
        self.stops = ev.stop_ids(self.model, self.processor.tokenizer)
        if quant is not None and not isinstance(quant, dict):
            quant = quant.to_dict() if hasattr(quant, "to_dict") else str(quant)
        self.description = {"engine": "transformers.generate", "precision": "bf16", "quantization": quant,
                            "batch_size": 1,
                            "decoding": "greedy", "model": src, "adapter": str(adapter) if adapter else None}

    def generate(self, messages, images, max_new_tokens=MAX_NEW_TOKENS):
        torch = self.torch
        batch = self.processor(text=[ev.prompt_text(self.processor, messages)], images=images,
                               return_tensors="pt").to(self.model.device)
        n_prompt = int(batch["input_ids"].shape[1])
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.inference_mode():
            gen = self.model.generate(**batch, max_new_tokens=max_new_tokens, do_sample=False,
                                      pad_token_id=self.processor.tokenizer.pad_token_id)
        torch.cuda.synchronize()
        secs = time.perf_counter() - t0
        new = gen[0, n_prompt:].tolist()
        n_new, truncated = ev.answer_length(new, self.stops)
        raw = self.processor.decode(new, skip_special_tokens=True)
        return Generation(raw, n_prompt, n_new, truncated, round(secs, 4))

    def close(self):
        del self.model
        self.torch.cuda.empty_cache()


def request(item, s):
    """(messages, images) for one item: system and user turns only, images resized as in training."""
    return ev.build(item, [], REPO / s["data_dir"], REPO / s["items_dir"])


def answer_items(engine, items, s, max_new_tokens=MAX_NEW_TOKENS, log=print, build=request):
    """One scored row per item, answered one request at a time by engine."""
    rows = []
    for i, item in enumerate(items, 1):
        g = engine.generate(*build(item, s), max_new_tokens=max_new_tokens)
        res = scoring.score(item, g.raw, s["style"])
        rows.append({"id": item["id"], "answer_type": item["answer_type"], "raw": g.raw,
                     "answer": list(parsed_answer(g.raw)), "pred": res["pred"],
                     **{k: bool(res[k]) for k in ("json_valid", "schema_valid", "correct", "abstained")},
                     **{k: v for k, v in asdict(g).items() if k != "raw"}})
        if log:
            log(f"  {i}/{len(items)} {'ok ' if res['correct'] else 'BAD'} {g.new_tokens} tok {g.seconds:.2f}s")
    return rows


def parsed_answer(raw):
    """The answer an output commits to, normalised: ("invalid",), ("null",), ("num", x) or ("text", s)."""
    out = scoring.extract_json(raw)
    if not isinstance(out, dict):
        return ("invalid",)
    a = out.get("answer")
    if a is None or (isinstance(a, str) and a.strip().lower() in ("", "null", "none")):
        return ("null",)
    if isinstance(a, bool):  # JSON true/false: a yes/no answer, not the number 1/0
        return ("text", "yes" if a else "no")
    if isinstance(a, (int, float)):
        return ("num", float(a))
    x = CT.num(a)
    return ("num", x) if x is not None else ("text", CT.norm(a))


def same_answer(a, b):
    """Whether two parsed answers (from parsed_answer, as tuples or lists) commit to the same answer.

    Numbers must agree to float precision: "1,390", "$1390" and 1390.0 are one answer; 5.45 and 5.46 are two.
    The scorer's 5% tolerance is a statement about gold labels, not about whether two models said the same thing.
    """
    a, b = tuple(a), tuple(b)
    if a[0] != b[0]:
        return False
    if a[0] == "num":
        return math.isclose(a[1], b[1], rel_tol=1e-9, abs_tol=1e-12)
    return a == b
