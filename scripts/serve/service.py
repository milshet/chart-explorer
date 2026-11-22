"""The request path of the endpoint (task 4.3), with no web framework in it: app.py only moves bytes in and out.

  POST /v1/answer  {"question": str, "image_base64": str}  ->  200 envelope, or 4xx {"error": {code, message}}
  GET  /health     the merge manifest of the checkpoint being served: is the thing that is up the thing measured?
  GET  /metrics    request, retry, fallback and refusal counters

The envelope carries the contract object under "result" and everything about how it was produced beside it:

  {"result": {...contract...}, "degraded": false, "attempts": 1, "retry_errors": [],
   "model": {"run", "status", "adapter_sha256"}, "usage": {...}, "timing": {...}}

Refusals, each with its own code and counter:
  - caller-supplied text that would enter the prompt ("context", "table", "messages", ...) is refused, not
    ignored: accepting it hands every caller the injection surface the red-team (4.5) attacks through pixels,
    only easier. The question is the only text a caller controls, and it goes into the training template.
  - unknown fields, a missing or over-long question, a body that is not a JSON object.
  - images: not base64, not decodable, a format other than PNG/JPEG/WebP, over MAX_IMAGE_BYTES, over
    MAX_DECODED_PIXELS (checked from the header, before decoding), or smaller than the model's 28-pixel patch.

The prompt is the training prompt: the same system turn and the same "Question: ...\\n\\nRespond with JSON: ..."
user turn (checked against every record in the dataset by the tests), and the image is resized exactly as in
training and evaluation. Generation is serialised: one model, batch 1, one request at a time (decision 12).
"""
import base64
import binascii
import io
import json
import re
import sys
import threading
import time
from collections import Counter
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1]
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
from serve import decoding  # noqa: E402

MAX_PIXELS = 800 * 800                  # smoke_test.MAX_PIXELS: the training and evaluation image size
MIN_SIDE = 28                           # one Qwen2.5-VL patch
MAX_IMAGE_BYTES = 8 * 2**20
MAX_DECODED_PIXELS = 40_000_000         # refused before decoding: a decompression bomb never reaches memory
MAX_BODY_BYTES = MAX_IMAGE_BYTES * 4 // 3 + 64 * 1024   # base64 inflates by 4/3; room for the JSON around it
MAX_QUESTION_CHARS = 500
FORMATS = ("PNG", "JPEG", "WEBP")
FIELDS = ("question", "image_base64")
PROMPT_FIELDS = ("context", "table", "data_table", "input_text", "messages", "system", "system_prompt", "prompt",
                 "instructions", "history")  # caller text that would reach the prompt: refused by name

SYSTEM = {  # the system turn of the training data, per output style
    "cot": "You are a chart analyst. Answer questions about the chart image provided.\n"
           "Return ONLY valid JSON matching the schema. Write the evidence and rationale first, then the answer.\n"
           'If the chart cannot answer the question, set "answer" to null and explain in "rationale".',
    "answer_only": "You are a chart analyst. Answer questions about the chart image provided.\n"
                   'Return ONLY valid JSON matching the schema. If the chart cannot answer the question,\n'
                   'set "answer" to null.',
}
STYLE = {"reasoning": "cot", "answer_only": "answer_only"}  # train_config targets -> output style


class Refusal(Exception):
    def __init__(self, status, code, message):
        super().__init__(message)
        self.status, self.code, self.message = status, code, message


def build_messages(question, style):
    """System and user turns exactly as in training; the image itself travels separately."""
    return [{"role": "system", "content": SYSTEM[style]},
            {"role": "user", "content": [{"type": "image", "image": "request"},
                                         {"type": "text", "text": f"Question: {question}\n\n{decoding.SCHEMA_LINE[style]}"}]}]


def prepare_image(img):
    """RGB, then shrunk to MAX_PIXELS exactly as smoke_test.load_image does for training and evaluation."""
    from PIL import Image
    img = img.convert("RGB")
    w, h = img.size
    if w * h > MAX_PIXELS:
        s = (MAX_PIXELS / (w * h)) ** 0.5
        img = img.resize((max(28, int(w * s)), max(28, int(h * s))), Image.BICUBIC)
    return img


def decode_image(b64):
    from PIL import Image
    if not isinstance(b64, str) or not b64:
        raise Refusal(400, "image_missing", '"image_base64" must be a non-empty base64 string')
    if len(b64) > MAX_IMAGE_BYTES * 4 // 3 + 4:
        raise Refusal(413, "image_too_large", f"the image is over {MAX_IMAGE_BYTES // 2**20} MB")
    try:
        data = base64.b64decode(b64, validate=True)
    except (binascii.Error, ValueError):
        raise Refusal(400, "image_not_base64", '"image_base64" is not valid base64') from None
    if len(data) > MAX_IMAGE_BYTES:
        raise Refusal(413, "image_too_large", f"the image is over {MAX_IMAGE_BYTES // 2**20} MB")
    try:
        img = Image.open(io.BytesIO(data))
    except Exception:  # PIL raises several unrelated types for undecodable bytes
        raise Refusal(400, "image_undecodable", "the image could not be decoded") from None
    if img.format not in FORMATS:
        raise Refusal(415, "image_format", f"image format {img.format} is not accepted; send {', '.join(FORMATS)}")
    w, h = img.size
    if w * h > MAX_DECODED_PIXELS:
        raise Refusal(413, "image_too_many_pixels", f"{w}x{h} is over {MAX_DECODED_PIXELS:,} pixels")
    if min(w, h) < MIN_SIDE:
        raise Refusal(400, "image_too_small", f"{w}x{h}: both sides must be at least {MIN_SIDE} pixels")
    try:
        img.load()
    except Exception:
        raise Refusal(400, "image_undecodable", "the image could not be decoded") from None
    return prepare_image(img)


def parse_request(body):
    """(question, image) from the raw request body, or a Refusal."""
    if len(body) > MAX_BODY_BYTES:
        raise Refusal(413, "body_too_large", f"the request body is over {MAX_BODY_BYTES // 2**20} MB")
    try:
        req = json.loads(body)
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise Refusal(400, "invalid_json", "the request body is not JSON") from None
    if not isinstance(req, dict):
        raise Refusal(400, "invalid_json", "the request body must be a JSON object")
    prompt_text = sorted(k for k in req if k.lower() in PROMPT_FIELDS)
    if prompt_text:
        raise Refusal(400, "caller_text_refused",
                      f"{', '.join(prompt_text)} refused: the endpoint takes no caller text besides the question, "
                      "so nothing a caller sends can rewrite the prompt")
    unknown = sorted(set(req) - set(FIELDS))
    if unknown:
        raise Refusal(400, "unknown_field", f"unknown field(s): {', '.join(unknown)}; send only {', '.join(FIELDS)}")
    return parse_question(req.get("question")), decode_image(req.get("image_base64"))


def parse_question(q):
    """The question on one line: a caller cannot add turns or blank lines to the template.

    Only line breaks are replaced; other spacing is kept, because training questions contain it (a double space
    in Eval-C) and the served prompt must be byte-identical to the training one.
    """
    if not isinstance(q, str) or not q.strip():
        raise Refusal(400, "question_missing", '"question" must be a non-empty string')
    q = re.sub(r"[ \t]*[\r\n\v\f\x85\u2028\u2029]+[ \t]*", " ", q).strip()
    if len(q) > MAX_QUESTION_CHARS:
        raise Refusal(400, "question_too_long", f"the question is over {MAX_QUESTION_CHARS} characters")
    return q


def served_style(manifest, targets=None):
    """The output style the served model was trained for: from --targets, else the adapter's train_config.json."""
    if targets is None:
        cfg_file = Path(manifest.get("adapter", "")) / "train_config.json"
        if not cfg_file.is_file():
            raise SystemExit(f"cannot tell what the served model was trained to output: no {cfg_file}; pass --targets")
        targets = json.loads(cfg_file.read_text(encoding="utf-8"))["targets"]
    return STYLE[targets]


class Service:
    def __init__(self, engine, manifest, style, allow_unverified=False, max_attempts=decoding.MAX_ATTEMPTS,
                 max_new_tokens=512, serving_commit=None):
        status = manifest.get("status")
        if status != "VERIFIED" and not allow_unverified:
            raise SystemExit(f"the checkpoint's merge manifest says {status!r}, not VERIFIED (task 4.1): not serving it")
        self.engine, self.manifest, self.style = engine, manifest, style
        self.max_attempts, self.max_new_tokens = max_attempts, max_new_tokens
        self.serving_commit, self.started = serving_commit, time.time()
        self.lock = threading.Lock()  # one model, batch 1: requests are answered one at a time
        self.counts, self.refused, self.errors_seen = Counter(), Counter(), Counter()

    def model_info(self):
        m = self.manifest
        return {"run": m.get("run"), "status": m.get("status"), "adapter_sha256": m.get("adapter_sha256")}

    def handle(self, body):
        """(HTTP status, JSON-serialisable payload) for one POST /v1/answer body."""
        t0 = time.perf_counter()
        try:
            question, image = parse_request(body)
        except Refusal as r:
            with self.lock:
                self.counts["requests"] += 1
                self.counts["refused"] += 1
                self.refused[r.code] += 1
            return r.status, {"error": {"code": r.code, "message": r.message}}
        with self.lock:
            self.counts["requests"] += 1
            try:
                out = decoding.answer(self.engine, build_messages(question, self.style), [image], self.style,
                                      self.max_attempts, self.max_new_tokens)
            except Exception as e:  # the engine failed (e.g. out of memory): an error, not an abstention
                self.counts["engine_errors"] += 1
                return 500, {"error": {"code": "engine_error", "message": f"{type(e).__name__}: {e}"}}
            self.count(out)
        failed = [a["errors"] for a in out.attempts if a["errors"]]
        return 200, {
            "result": out.result, "degraded": out.degraded, "attempts": len(out.attempts), "retry_errors": failed,
            "model": self.model_info(),
            "usage": {"prompt_tokens": out.attempts[0]["prompt_tokens"],
                      "new_tokens": sum(a["new_tokens"] for a in out.attempts)},
            "timing": {"generate_s": round(sum(a["seconds"] for a in out.attempts), 4),
                       "total_s": round(time.perf_counter() - t0, 4)}}

    def count(self, out):
        c = self.counts
        c["answered"] += 1
        c["attempts"] += len(out.attempts)
        if len(out.attempts) > 1:
            c["retried"] += 1
            c["retries"] += len(out.attempts) - 1
        if not out.attempts[0]["errors"]:
            c["first_attempt_valid"] += 1
        elif not out.degraded:
            c["recovered_by_retry"] += 1
        if out.degraded:
            c["degraded"] += 1
        elif out.result.get("answer") is None:
            c["model_abstained"] += 1  # the model's own refusal; fallbacks are counted apart, never here
        for a in out.attempts:
            for e in a["errors"]:
                self.errors_seen[e.split(":")[0][:60]] += 1

    def metrics(self):
        with self.lock:
            c = dict(self.counts)
            answered = c.get("answered", 0)
            rate = lambda k: round(c.get(k, 0) / answered, 4) if answered else None  # noqa: E731
            return {"counters": {k: c.get(k, 0) for k in (
                        "requests", "answered", "first_attempt_valid", "retried", "retries", "recovered_by_retry",
                        "degraded", "model_abstained", "refused", "engine_errors", "attempts")},
                    "rates": {"retried": rate("retried"), "degraded": rate("degraded"),
                              "first_attempt_valid": rate("first_attempt_valid"),
                              "model_abstained": rate("model_abstained")},
                    "refused_by_code": dict(self.refused), "validation_errors_seen": dict(self.errors_seen),
                    "max_attempts": self.max_attempts}

    def health(self):
        m = self.manifest
        return {"status": "ok" if m.get("status") == "VERIFIED" else "serving_unverified_checkpoint",
                "model": {**self.model_info(), "merged": m.get("merged"), "base": m.get("base_check", {}).get("base"),
                          "base_revision": m.get("base_check", {}).get("base_revision"),
                          "merged_files_sha256": m.get("merged_files_sha256"), "merge_commit": m.get("git_commit"),
                          "answer_check": m.get("answer_check")},
                "serving": {"engine": "transformers.generate", "precision": "bf16", "batch_size": 1,
                            "decoding": "greedy", "style": self.style, "max_new_tokens": self.max_new_tokens,
                            "max_attempts": self.max_attempts, "git_commit": self.serving_commit,
                            "uptime_s": round(time.time() - self.started)}}
