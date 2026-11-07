from itertools import count
from platform import processor

import modal

MODEL_ID = "Qwen/Qwen3-VL-4B-Instruct"
DATASET_ID = "HuggingFaceM4/ChartQA"
app = modal.App("chartqa-qwen3-vl-4b")
image = (modal.Image.debian_slim(python_version="3.11")
         .pip_install("torch>=2.5", "torchvision", "transformers>=4.57", "accelerate>=1.0", "datasets>=3.0", "pillow", "peft>=0.16", "trl>=0.22", "bitsandbytes>=0.46", "scipy", "sentencepiece", "qwen-vl-utils", "huggingface_hub"))
artifacts = modal.Volume.from_name("chartqa-qwen3-artifacts", create_if_missing=True)
cache = modal.Volume.from_name("chartqa-qwen3-hf-cache", create_if_missing=True)


def normalize(x):
    return str(x).strip().lower().replace(',', '').replace('%', '').replace('$', '')


def relaxed_match(pred, refs):
    pred = normalize(pred)
    for ref in refs:
        gold = normalize(ref)
        if pred == gold:
            return True
        try:
            p, g = float(pred), float(gold)
            if abs(p - g) <= 0.05 * abs(g) if g != 0 else abs(p - g) <= 1e-9:
                return True
        except (TypeError, ValueError):
            pass
    return False




def score_chartqa(pred, refs, normalize_format=False):
    """ChartQA-style: 5% relative numeric tolerance; exact text otherwise.

    Optional format-normalization is diagnostic only, NOT benchmark comparable.
    """
    import re
    import math
    def clean(value):
        value = str(value).strip().lower().rstrip(".").strip()
        if normalize_format:
            # Only outer list wrappers and repeated whitespace are removed.
            if len(value) > 1 and value[0] == "[" and value[-1] == "]":
                value = value[1:-1].strip()
            value = " ".join(value.split())
        return value
    def as_number(s):
        compact = s.replace(",", "").replace("$", "").strip()
        if compact.endswith("%"):
            compact = compact[:-1].strip()
        if not re.fullmatch(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)", compact):
            return None
        number = float(compact)
        return number if math.isfinite(number) else None
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


# CPU-only job: reuses previously saved predictions, does not load Qwen or request a GPU.
@app.function(image=modal.Image.debian_slim(python_version="3.11"), cpu=1,
              volumes={"/artifacts": artifacts}, timeout=300)
def rescore_saved(count: int = 100, adapter: bool = False):
    import json
    from pathlib import Path
    name = "finetuned" if adapter else "baseline"
    source = Path(f"/artifacts/{name}_val_{count}.json")
    if not source.exists():
        raise FileNotFoundError(f"No saved predictions at {source}; run the corresponding inference first.")
    data = json.loads(source.read_text())
    rows = data["results"]
    if not rows:
        raise ValueError("No predictions in file")
    strict_correct, fmt_correct, changes = 0, 0, []
    by_group = {}
    for row in rows:
        pred = row["prediction"]
        refs = row["reference"]
        old = bool(row["correct"])
        # if row["index"] == 84:
        #     print("\n--- Question 85 Debug ---")
        #     print("Prediction:", repr(pred))
        #     print("Reference:", repr(refs))
        #     print("Original:", old)
        #     print("Old scorer:", relaxed_match(pred, refs))
        #     print("New scorer:", score_chartqa(pred, refs))
        strict = score_chartqa(pred, refs)
        fmt = score_chartqa(pred, refs, normalize_format=True)
        strict_correct += strict
        fmt_correct += fmt
        group = row.get("human_or_machine", "unknown")
        grp = by_group.setdefault(group, {"count": 0, "strict_correct": 0})
        grp["count"] += 1
        grp["strict_correct"] += strict
        if old != strict or strict != fmt:
            changes.append({"index": row["index"], "question": row["question"],
                            "prediction": pred, "reference": refs,
                            "original": old, "strict": strict, "format_normalized": fmt})
    metrics = {
        "source": str(source), "count": len(rows),
        "original_accuracy": data.get("relaxed_accuracy"),
        "strict_chartqa_style_accuracy": strict_correct / len(rows),
        "format_normalized_diagnostic_accuracy": fmt_correct / len(rows),
        "note": "The diagnostic format-normalized score is NOT official ChartQA accuracy.",
        "per_group": {k: {**v, "accuracy": v["strict_correct"] / v["count"]}
                      for k, v in by_group.items()},
        "disagreements": changes,
    }
    out = Path(f"/artifacts/{name}_val_{count}_rescored.json")
    out.write_text(json.dumps(metrics, indent=2))
    artifacts.commit()
    print(f"Original: {metrics['original_accuracy']:.2%} | ChartQA-style: {metrics['strict_chartqa_style_accuracy']:.2%} | Format diagnostic: {metrics['format_normalized_diagnostic_accuracy']:.2%}")
    print(f"Items with scoring disagreements: {len(changes)}")
    for ch in changes[:25]:
        print(f"#{ch['index']+1} old={ch['original']} strict={ch['strict']} format={ch['format_normalized']} pred={ch['prediction']!r} ref={ch['reference']!r}")
    print(f"Saved: {out}")
    return {"path": str(out), "original": metrics["original_accuracy"],
            "strict": metrics["strict_chartqa_style_accuracy"],
            "diagnostic": metrics["format_normalized_diagnostic_accuracy"]}


def get_model(adapter=False):
    import torch
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration, BitsAndBytesConfig
    from peft import PeftModel

    if adapter:
        model = PeftModel.from_pretrained(
            model,
            "/artifacts/runs/chartqa_5k/adapter"
        )

    config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = Qwen3VLForConditionalGeneration.from_pretrained(MODEL_ID, quantization_config=config, device_map="auto", torch_dtype=torch.bfloat16, attn_implementation="sdpa")
    if adapter:
        model = PeftModel.from_pretrained(model, "/artifacts/adapter_pilot_100")
    model.eval()
    return model, processor


def answer(model, processor, item):
    import torch
    messages = [{"role": "user", "content": [{"type": "image", "image": item["image"].convert("RGB")}, {"type": "text", "text": item["query"] + "\nAnswer with only the final short answer."}]}]
    inputs = processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt")
    inputs.pop("token_type_ids", None)
    inputs = inputs.to(model.device)
    with torch.inference_mode():
        generated = model.generate(**inputs, max_new_tokens=48, do_sample=False)
    return processor.decode(generated[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True).strip()


@app.function(image=image, gpu=["L4", "A10", "A100"], cpu=4, memory=16384, timeout=3600, volumes={"/artifacts": artifacts, "/cache": cache}, env={"HF_HOME": "/cache", "HF_HUB_DISABLE_TELEMETRY": "1"})
def run_inference(count: int = 1, adapter: bool = False):
    import json
    from pathlib import Path
    from datasets import load_dataset
    ds = load_dataset(DATASET_ID, split="val")
    count = min(count, len(ds))
    model, processor = get_model(adapter)
    results = []
    for idx in range(count):
        sample = ds[idx]
        prediction = answer(model, processor, sample)
        result = {"index": idx, "question": sample["query"], "reference": sample["label"], "prediction": prediction, "correct": score_chartqa(prediction, sample["label"]), "human_or_machine": str(sample["human_or_machine"])}
        results.append(result)
        print(f"[{idx+1}/{count}] {result['correct']} Q: {result['question']} Pred: {prediction} Ref: {sample['label']}", flush=True)
    accuracy = sum(r["correct"] for r in results) / len(results)
    name = "finetuned" if adapter else "baseline"
    path = Path(f"/artifacts/{name}_val_{count}.json")
    path.write_text(json.dumps({"model": MODEL_ID, "split": "val", "count": count, "relaxed_accuracy": accuracy, "results": results}, indent=2))
    artifacts.commit()
    print(f"{name} relaxed accuracy: {accuracy:.2%}; output: {path}")
    return {"accuracy": accuracy, "saved": str(path)}


@app.function(image=image, gpu=["L4", "A10", "A100"], cpu=4, memory=24000, timeout=86400, volumes={"/artifacts": artifacts, "/cache": cache}, env={"HF_HOME": "/cache", "HF_HUB_DISABLE_TELEMETRY": "1"})
def train(count: int = 500, steps: int = 100):
    import torch
    from transformers import TrainerCallback
    from datasets import load_dataset
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration, BitsAndBytesConfig
    from peft import LoraConfig, prepare_model_for_kbit_training
    from trl import SFTConfig, SFTTrainer
    #ds = load_dataset(DATASET_ID, split=f"train[:{count}]")
    ds = load_dataset(DATASET_ID, split="train")
    ds = ds.shuffle(seed=42).select(range(min(count, len(ds))))
    def prepare(row):
        return {"image": row["image"].convert("RGB"), "prompt": [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": row["query"] + "\nAnswer with only the final short answer."}]}], "completion": [{"role": "assistant", "content": [{"type": "text", "text": row["label"][0]}]}]}
    ds = ds.map(prepare, remove_columns=ds.column_names)
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16)
    model = Qwen3VLForConditionalGeneration.from_pretrained(MODEL_ID, quantization_config=quant, device_map="auto", torch_dtype=torch.bfloat16, attn_implementation="sdpa")
    model = prepare_model_for_kbit_training(model)
    model.config.use_cache = False
    lora = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none", target_modules="all-linear", task_type="CAUSAL_LM")
    #args = SFTConfig(output_dir="/artifacts/checkpoints", per_device_train_batch_size=1, gradient_accumulation_steps=8, max_steps=steps, learning_rate=1e-4, logging_steps=10, save_steps=50, save_total_limit=2, bf16=True, gradient_checkpointing=True, max_length=None, remove_unused_columns=False, report_to="none", dataset_num_proc=1,completion_only_loss=True,save_strategy="no")
    
    args = SFTConfig(
        output_dir="/artifacts/runs/chartqa_5k/checkpoints",

        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,

        max_steps=steps,
        learning_rate=5e-5,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,

        logging_steps=10,

        save_strategy="steps",
        save_steps=125,
        save_total_limit=3,
        save_safetensors=True,

        completion_only_loss=True,
        bf16=True,
        gradient_checkpointing=True,

        max_length=None,
        remove_unused_columns=False,

        dataset_num_proc=1,
        seed=42,
        data_seed=42,
        report_to="none",
    )



    class ModalCheckpointCallback(TrainerCallback):
        def on_save(self, args, state, control, **kwargs):
            artifacts.commit()
            print(
                f"Checkpoint persisted at step {state.global_step}",
                flush=True
            )
            return control

    
    trainer = SFTTrainer(model=model, args=args, train_dataset=ds, processing_class=processor, peft_config=lora)
    trainer.add_callback(ModalCheckpointCallback())
    trainer.train()


    adapter_path = "/artifacts/runs/chartqa_5k/adapter"

    trainer.save_model(adapter_path)
    processor.save_pretrained(adapter_path)

    artifacts.commit()

    print(f"Saved LoRA adapter to {adapter_path}")

@app.local_entrypoint()
def main(stage: str = "smoke", count: int = 100, steps: int = 100):
    if stage == "smoke":
        print(run_inference.remote(count=1, adapter=False))
    elif stage == "baseline":
        print(run_inference.remote(count=count, adapter=False))
    elif stage == "rescore":
        print(rescore_saved.remote(count=count, adapter=False))
    elif stage == "rescore_finetuned":
        print(rescore_saved.remote(count=count, adapter=True))
    elif stage == "train":
        train.remote(count=count, steps=steps)
    elif stage == "evaluate":
        print(run_inference.remote(count=count, adapter=True))
    else:
        raise ValueError("stage must be smoke, baseline, rescore, train, evaluate, or rescore_finetuned")
