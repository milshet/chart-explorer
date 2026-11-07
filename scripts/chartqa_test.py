"""Held-out ChartQA test evaluation: pretrained Qwen3-VL-4B vs 10K checkpoint 1000.

Run each test pass only after model-selection decisions are frozen.
The relaxed scorer is reproduced from the user's prior chartqa_10k.py; it is
NOT independently certified to be equivalent to the official ChartQA evaluator.
"""
import json
import math
import re
from pathlib import Path

import modal

MODEL_ID = "Qwen/Qwen3-VL-4B-Instruct"
DATASET_ID = "HuggingFaceM4/ChartQA"
ROOT = "/artifacts/runs/chartqa_10k_lr3e5"
SELECTED_CHECKPOINT = "1000"
SELECTED_ADAPTER_PATH = f"{ROOT}/checkpoints/checkpoint-{SELECTED_CHECKPOINT}"
OUT_DIR = f"{ROOT}/final_test"

app = modal.App("chartqa-qwen3-vl-4b-heldout-test")
image = (modal.Image.debian_slim(python_version="3.11")
         .pip_install("torch>=2.5", "torchvision", "transformers>=4.57", "accelerate>=1.0",
                      "datasets>=3.0", "pillow", "peft>=0.16", "trl>=0.22",
                      "bitsandbytes>=0.46", "scipy", "sentencepiece",
                      "qwen-vl-utils", "huggingface_hub"))
artifacts = modal.Volume.from_name("chartqa-qwen3-artifacts", create_if_missing=False)
cache = modal.Volume.from_name("chartqa-qwen3-hf-cache", create_if_missing=False)
VOLUMES = {"/artifacts": artifacts, "/cache": cache}
ENV = {"HF_HOME": "/cache", "HF_HUB_DISABLE_TELEMETRY": "1"}


def score_chartqa(pred, refs):
    """Same local 5%-relative-numeric-tolerance scorer as the 10K validation run."""
    def clean(value):
        return str(value).strip().lower().rstrip(".").strip()

    def as_number(s):
        compact = s.replace(",", "").replace("$", "").strip()
        if compact.endswith("%"):
            compact = compact[:-1].strip()
        if not re.fullmatch(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)", compact):
            return None
        value = float(compact)
        return value if math.isfinite(value) else None

    if not isinstance(refs, list):
        refs = [refs]
    p = clean(pred)
    for ref in refs:
        g = clean(ref)
        pn, gn = as_number(p), as_number(g)
        if pn is not None and gn is not None:
            if abs(pn - gn) <= 0.05 * abs(gn) if gn != 0 else pn == 0:
                return True
        elif p == g:
            return True
    return False


def diagnostic(row):
    """Supplementary heuristics. Do not use them to change the relaxed score."""
    question = row["question"].lower()
    pred = str(row["prediction"]).strip()
    refs = row["reference"] if isinstance(row["reference"], list) else [row["reference"]]
    result = {"year_answer_mismatch": False, "possible_100x_scale_mismatch": False}
    if ("year" in question or "when" in question) and any(re.fullmatch(r"\d{4}", str(g).strip()) for g in refs):
        result["year_answer_mismatch"] = not any(pred == str(g).strip() for g in refs)
    try:
        number = float(pred.replace(",", "").replace("%", "").replace("$", ""))
    except ValueError:
        return result
    for g in refs:
        try:
            gold = float(str(g).strip().replace(",", "").replace("%", "").replace("$", ""))
        except ValueError:
            continue
        if gold and number and (abs(number / gold - 100) < 5 or abs(number / gold - 0.01) < 0.0005):
            result["possible_100x_scale_mismatch"] = True
    return result


def model_and_processor(use_adapter):
    import torch
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration, BitsAndBytesConfig
    from peft import PeftModel

    processor = AutoProcessor.from_pretrained(MODEL_ID)
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_ID, quantization_config=quantization,
        device_map="auto", torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    if use_adapter:
        if not Path(SELECTED_ADAPTER_PATH).exists():
            raise FileNotFoundError(f"Missing selected adapter checkpoint: {SELECTED_ADAPTER_PATH}")
        model = PeftModel.from_pretrained(model, SELECTED_ADAPTER_PATH)
    model.eval()
    return model, processor


def predict(model, processor, sample):
    import torch
    messages = [{"role": "user", "content": [
        {"type": "image", "image": sample["image"].convert("RGB")},
        {"type": "text", "text": sample["query"] + "\nAnswer with only the final short answer."},
    ]}]
    inputs = processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True,
        return_dict=True, return_tensors="pt",
    )
    inputs.pop("token_type_ids", None)
    inputs = inputs.to(model.device)
    with torch.inference_mode():
        generated = model.generate(**inputs, max_new_tokens=48, do_sample=False)
    return processor.decode(
        generated[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True,
    ).strip()


@app.function(image=image, gpu="L4", cpu=4, memory=16384, timeout=21600,
              volumes=VOLUMES, env=ENV)
def run_test(kind: str, count: int = 2500):
    from datasets import load_dataset

    if kind not in {"baseline", "ckpt1000"}:
        raise ValueError("kind must be baseline or ckpt1000")
    ds = load_dataset(DATASET_ID, split="test")
    if count <= 0:
        raise ValueError("count must be positive")
    count = min(count, len(ds))
    use_adapter = (kind == "ckpt1000")
    output = Path(f"{OUT_DIR}/{kind}_test_{count}.json")
    if output.exists():
        raise FileExistsError(f"Test predictions already exist: {output}; preserve them instead of overwriting")

    model, processor = model_and_processor(use_adapter)
    rows = []
    for idx in range(count):
        item = ds[idx]
        prediction = predict(model, processor, item)
        row = {
            "index": idx, "question": item["query"],
            "reference": item["label"], "prediction": prediction,
            "human_or_machine": str(item["human_or_machine"]),
            "correct": score_chartqa(prediction, item["label"]),
        }
        rows.append(row)
        if (idx + 1) % 100 == 0 or (idx + 1) == count:
            print(f"[{kind} {idx + 1}/{count}] relaxed accuracy = {sum(r['correct'] for r in rows)/(idx+1):.2%}", flush=True)

    result = {
        "model": MODEL_ID,
        "dataset": DATASET_ID,
        "split": "test",
        "count": count,
        "kind": kind,
        "adapter_path": SELECTED_ADAPTER_PATH if use_adapter else None,
        "metric": "local_chartqa_style_relaxed_5pct_not_officially_verified",
        "relaxed_accuracy": sum(r["correct"] for r in rows) / count,
        "results": rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    artifacts.commit()
    print(f"Saved {kind} test results: {result['relaxed_accuracy']:.2%} at {output}", flush=True)
    return {"accuracy": result["relaxed_accuracy"], "file": str(output)}


@app.function(image=modal.Image.debian_slim(python_version="3.11"), cpu=1,
              volumes={"/artifacts": artifacts}, timeout=300)
def compare_test(count: int = 2500):
    a = Path(f"{OUT_DIR}/baseline_test_{count}.json")
    b = Path(f"{OUT_DIR}/ckpt1000_test_{count}.json")
    if not a.exists() or not b.exists():
        raise FileNotFoundError(f"Both test prediction files are required: {a}, {b}")
    base = json.loads(a.read_text())
    tuned = json.loads(b.read_text())
    br = {r["index"]: r for r in base["results"]}
    tr = {r["index"]: r for r in tuned["results"]}
    if br.keys() != tr.keys() or len(br) != count:
        raise ValueError("Different evaluation indices; cannot compare")
    for idx in br:
        if br[idx]["question"] != tr[idx]["question"] or br[idx]["reference"] != tr[idx]["reference"]:
            raise ValueError(f"Different underlying test example at index {idx}")

    group_counts = {}
    fixed = regressed = 0
    diagnostics = {"baseline": {"year_answer_mismatch": 0, "possible_100x_scale_mismatch": 0},
                   "ckpt1000": {"year_answer_mismatch": 0, "possible_100x_scale_mismatch": 0}}
    changed_examples = []
    for idx in sorted(br):
        old, new = br[idx], tr[idx]
        old_ok = score_chartqa(old["prediction"], old["reference"])
        new_ok = score_chartqa(new["prediction"], new["reference"])
        fixed += int(not old_ok and new_ok)
        regressed += int(old_ok and not new_ok)
        group = old["human_or_machine"]
        entry = group_counts.setdefault(group, {"total": 0, "baseline_correct": 0, "ckpt1000_correct": 0})
        entry["total"] += 1
        entry["baseline_correct"] += int(old_ok)
        entry["ckpt1000_correct"] += int(new_ok)
        for label, row in (("baseline", old), ("ckpt1000", new)):
            for key, val in diagnostic(row).items():
                diagnostics[label][key] += int(val)
        if old_ok != new_ok:
            changed_examples.append({"index": idx, "question": new["question"],
                                     "reference": new["reference"],
                                     "baseline_prediction": old["prediction"],
                                     "tuned_prediction": new["prediction"],
                                     "baseline_correct": old_ok, "ckpt1000_correct": new_ok})

    summary = {
        "split": "test", "count": count,
        "metric": "local_chartqa_style_relaxed_5pct_not_officially_verified",
        "baseline_accuracy": sum(score_chartqa(r["prediction"], r["reference"]) for r in br.values()) / count,
        "ckpt1000_accuracy": sum(score_chartqa(r["prediction"], r["reference"]) for r in tr.values()) / count,
        "fixes": fixed, "regressions": regressed,
        "group_metrics": {g: {**v,
                              "baseline_accuracy": v["baseline_correct"] / v["total"],
                              "ckpt1000_accuracy": v["ckpt1000_correct"] / v["total"]}
                          for g, v in group_counts.items()},
        "diagnostics": diagnostics,
        "changed_examples": changed_examples,
    }
    out = Path(f"{OUT_DIR}/comparison_test_{count}.json")
    if out.exists():
        raise FileExistsError(f"Already exists: {out}")
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    artifacts.commit()
    print(json.dumps({k: v for k, v in summary.items() if k != "changed_examples"}, indent=2), flush=True)
    print(f"Full comparison saved to {out}", flush=True)
    return {"baseline_accuracy": summary["baseline_accuracy"], "ckpt1000_accuracy": summary["ckpt1000_accuracy"], "comparison": str(out)}


@app.local_entrypoint()
def main(stage: str = "compare", count: int = 2500):
    if stage == "baseline":
        return run_test.remote(kind="baseline", count=count)
    if stage == "finetuned":
        return run_test.remote(kind="ckpt1000", count=count)
    if stage == "compare":
        return compare_test.remote(count=count)
    raise ValueError("stage must be baseline, finetuned, or compare")
