#!/usr/bin/env python3
"""Serving-stack smoke test (task 3.0f): does vLLM, in its own venv, load a merged Qwen2.5-VL checkpoint and
answer one multimodal request with a contract-valid object?

Decided 2026-11-19 (docs/SERVING.md): --engine transformers, run from ~/venv, the locked training stack, so the
served model runs on exactly the stack every reported number was measured on. vLLM did not fit the per-user quota;
the vllm engine stays for a separate venv (SERVE_VENV unset activates ~/venv-serve), never ~/venv.

  M=/storage/tracka/users/learner01/merged/run_a_r16_smoke
  SERVE_VENV=$HOME/venv sbatch slurm/serve_smoke.sbatch --merge ~/runs/run_a_r16/adapter --merged $M --engine transformers
  SERVE_VENV=$HOME/venv sbatch slurm/serve_smoke.sbatch --model $M --engine transformers     # request only

Writes a JSON log (--log, default logs/serve_smoke.json): engine, package versions, the request, the raw
output, the scorer's verdict, load and generation seconds. Exit status 0 only when the output is json_valid,
schema_valid and correct on the probe item: "it ran" is not the bar, "it answered within the contract" is.
"""
import argparse
import json
import platform
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
import scoring  # noqa: E402  pure Python: imports nothing from the training stack

BASE = "Qwen/Qwen2.5-VL-3B-Instruct"
PROBE_ID = "val-two_col_83529-542"  # Run A and Run B both answer it correctly on val: a wrong answer here is the stack
PACKAGES = ("vllm", "torch", "transformers", "fastapi", "uvicorn", "pydantic", "peft", "pillow", "autoawq", "llmcompressor")


def probe(item_id=PROBE_ID, data=REPO / "data/v2.0/image_only/val.jsonl"):
    for line in open(data, encoding="utf-8"):
        r = json.loads(line)
        if r["id"] == item_id:
            return r
    raise SystemExit(f"{item_id} not in {data}")


def openai_messages(item, image_path):
    """The item's system and user turns in the OpenAI chat format vLLM's LLM.chat takes; image as a data URI."""
    import base64
    uri = "data:image/png;base64," + base64.b64encode(Path(image_path).read_bytes()).decode()
    user = [{"type": "image_url", "image_url": {"url": uri}} if c["type"] == "image" else c
            for c in item["messages"][1]["content"]]
    return [{"role": "system", "content": item["messages"][0]["content"]}, {"role": "user", "content": user}]


def versions():
    from importlib.metadata import PackageNotFoundError, version
    out = {"python": platform.python_version()}
    for p in PACKAGES:
        try:
            out[p] = version(p)
        except PackageNotFoundError:
            out[p] = None
    return out


def merge(adapter, merged):
    """bf16 base + adapter -> merged weights. Never 4-bit: folding deltas into quantised weights is neither model."""
    import torch
    from peft import PeftModel
    from transformers import AutoModelForImageTextToText, AutoProcessor
    model = AutoModelForImageTextToText.from_pretrained(BASE, dtype=torch.bfloat16, local_files_only=True)
    model = PeftModel.from_pretrained(model, adapter).merge_and_unload()
    model.save_pretrained(merged, safe_serialization=True)
    AutoProcessor.from_pretrained(adapter).save_pretrained(merged)
    return {"merged": str(merged), "adapter": str(adapter)}


def run_vllm(model, item, image_path, max_tokens):
    from vllm import LLM, SamplingParams
    t0 = time.monotonic()
    llm = LLM(model=str(model), max_model_len=4096, limit_mm_per_prompt={"image": 1},
              mm_processor_kwargs={"max_pixels": 800 * 800}, gpu_memory_utilization=0.6)
    load = time.monotonic() - t0
    t0 = time.monotonic()
    out = llm.chat(openai_messages(item, image_path), SamplingParams(temperature=0, max_tokens=max_tokens))
    return out[0].outputs[0].text, load, time.monotonic() - t0


def run_transformers(model, item, image_path, max_tokens):
    import torch
    from PIL import Image
    from transformers import AutoModelForImageTextToText, AutoProcessor
    t0 = time.monotonic()
    processor = AutoProcessor.from_pretrained(model, max_pixels=800 * 800)
    net = AutoModelForImageTextToText.from_pretrained(model, dtype=torch.bfloat16, device_map={"": 0})
    load = time.monotonic() - t0
    msgs = item["messages"][:2]
    text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    batch = processor(text=[text], images=[Image.open(image_path).convert("RGB")], return_tensors="pt").to(net.device)
    t0 = time.monotonic()
    gen = net.generate(**batch, max_new_tokens=max_tokens, do_sample=False)
    raw = processor.batch_decode(gen[:, batch["input_ids"].shape[1]:], skip_special_tokens=True)[0]
    return raw, load, time.monotonic() - t0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=BASE, help="merged checkpoint directory (or the base model id)")
    ap.add_argument("--merge", type=Path, help="adapter to merge into bf16 base weights first, written to --merged")
    ap.add_argument("--merged", type=Path)
    ap.add_argument("--engine", choices=("vllm", "transformers"), default="vllm")
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--log", type=Path, default=REPO / "logs" / "serve_smoke.json")
    args = ap.parse_args(argv)
    log = {"task": "3.0f", "versions": versions()}
    if args.merge:
        if not args.merged:
            ap.error("--merge needs --merged")
        log["merge"] = merge(args.merge.expanduser(), args.merged.expanduser())
        args.model = str(args.merged.expanduser())
    item = probe()
    image = REPO / "data/v2.0" / item["image"]
    run = run_vllm if args.engine == "vllm" else run_transformers
    raw, load_s, gen_s = run(args.model, item, image, args.max_tokens)
    verdict = scoring.score(item, raw, "cot")
    log.update({"engine": args.engine, "model": args.model, "probe_id": item["id"], "raw": raw,
                "verdict": {k: verdict[k] for k in ("json_valid", "schema_valid", "correct", "pred")},
                "load_seconds": round(load_s, 1), "generate_seconds": round(gen_s, 2)})
    ok = verdict["json_valid"] and verdict["schema_valid"] and verdict["correct"]
    log["status"] = "SERVE_SMOKE_OK" if ok else "SERVE_SMOKE_FAILED"
    args.log.parent.mkdir(parents=True, exist_ok=True)
    args.log.write_text(json.dumps(log, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(log, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
