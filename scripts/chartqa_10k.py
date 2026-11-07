"""Independent 10K QLoRA experiment; no modifications to the existing 5K run."""
import json
import math
import re
from pathlib import Path

import modal

MODEL_ID = "Qwen/Qwen3-VL-4B-Instruct"
DATASET_ID = "HuggingFaceM4/ChartQA"
EXPERIMENT = "chartqa_10k_lr3e5"
ROOT = f"/artifacts/runs/{EXPERIMENT}"

app = modal.App("chartqa-qwen3-vl-4b-10k")
image = (modal.Image.debian_slim(python_version="3.11")
         .pip_install("torch>=2.5", "torchvision", "transformers>=4.57", "accelerate>=1.0",
                      "datasets>=3.0", "pillow", "peft>=0.16", "trl>=0.22",
                      "bitsandbytes>=0.46", "scipy", "sentencepiece",
                      "qwen-vl-utils", "huggingface_hub"))
artifacts = modal.Volume.from_name("chartqa-qwen3-artifacts", create_if_missing=True)
cache = modal.Volume.from_name("chartqa-qwen3-hf-cache", create_if_missing=True)
VOLUMES = {"/artifacts": artifacts, "/cache": cache}
ENV = {"HF_HOME": "/cache", "HF_HUB_DISABLE_TELEMETRY": "1"}


def score_chartqa(pred, refs):
    """Same ChartQA-style relaxed scorer used in the user's earlier verified runs.

    Still a locally implemented metric, not a claim of byte-for-byte official equivalence.
    """
    def clean(value):
        return str(value).strip().lower().rstrip(".").strip()

    def as_number(s):
        compact = s.replace(",", "").replace("$", "").strip()
        if compact.endswith("%"):
            compact = compact[:-1].strip()
        if not re.fullmatch(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)", compact):
            return None
        v = float(compact)
        return v if math.isfinite(v) else None

    if not isinstance(refs, list):
        refs = [refs]
    p = clean(pred)
    for r in refs:
        g = clean(r)
        pn, gn = as_number(p), as_number(g)
        if pn is not None and gn is not None:
            if (abs(pn - gn) <= 0.05 * abs(gn)) if gn != 0 else (pn == 0):
                return True
        elif p == g:
            return True
    return False


def diagnostic_label(pred, refs):
    """Report potential errors separately; do not change the benchmark-style score."""
    if not isinstance(refs, list):
        refs = [refs]
    p = str(pred).strip().replace(",", "").replace("%", "")
    for ref in refs:
        g = str(ref).strip().replace(",", "").replace("%", "")
        try:
            pn, gn = float(p), float(g)
        except ValueError:
            continue
        if gn and pn and (abs(pn / gn - 100) < 5 or abs(pn / gn - 0.01) < 0.0005):
            return "possible_100x_scale_mismatch"
        # For explicit year questions use exact numeric answer, not 5% tolerance.
    return None


def load_model(adapter_path):
    import torch
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration, BitsAndBytesConfig
    from peft import PeftModel
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                               bnb_4bit_compute_dtype=torch.bfloat16,
                               bnb_4bit_use_double_quant=True)
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_ID, quantization_config=quant, device_map="auto",
        torch_dtype=torch.bfloat16, attn_implementation="sdpa")
    model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    return model, processor


def answer(model, processor, item):
    import torch
    messages = [{"role": "user", "content": [
        {"type": "image", "image": item["image"].convert("RGB")},
        {"type": "text", "text": item["query"] + "\nAnswer with only the final short answer."}]}]
    inputs = processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True,
        return_dict=True, return_tensors="pt")
    inputs.pop("token_type_ids", None)
    inputs = inputs.to(model.device)
    with torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=48, do_sample=False)
    return processor.decode(out[0][inputs["input_ids"].shape[-1]:],
                            skip_special_tokens=True).strip()


@app.function(image=image, gpu="L4", cpu=4, memory=24000, timeout=86400,
              volumes=VOLUMES, env=ENV)
def train(count: int = 10000, steps: int = 1250):
    import torch
    from datasets import load_dataset
    from transformers import (AutoProcessor, Qwen3VLForConditionalGeneration,
                              BitsAndBytesConfig, TrainerCallback)
    from peft import LoraConfig, prepare_model_for_kbit_training
    from trl import SFTConfig, SFTTrainer

    # Independent run from the original model, not continued from the 5K adapter.
    ds = load_dataset(DATASET_ID, split="train")
    ds = ds.shuffle(seed=42).select(range(min(count, len(ds))))

    def prepare(row):
        return {
            "image": row["image"].convert("RGB"),
            "prompt": [{"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": row["query"] + "\nAnswer with only the final short answer."}]}],
            "completion": [{"role": "assistant", "content": [
                {"type": "text", "text": row["label"][0]}]}],
        }

    ds = ds.map(prepare, remove_columns=ds.column_names)
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                               bnb_4bit_use_double_quant=True,
                               bnb_4bit_compute_dtype=torch.bfloat16)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_ID, quantization_config=quant, device_map="auto",
        torch_dtype=torch.bfloat16, attn_implementation="sdpa")
    model = prepare_model_for_kbit_training(model)
    model.config.use_cache = False
    lora = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
                      target_modules="all-linear", task_type="CAUSAL_LM")

    args = SFTConfig(
        output_dir=f"{ROOT}/checkpoints",
        per_device_train_batch_size=1, gradient_accumulation_steps=8,
        max_steps=steps, learning_rate=3e-5,
        lr_scheduler_type="cosine", warmup_steps=38,
        logging_steps=20, save_strategy="steps", save_steps=250,
        save_total_limit=5, completion_only_loss=True,
        bf16=True, gradient_checkpointing=True, max_length=None,
        remove_unused_columns=False, dataset_num_proc=1,
        seed=42, data_seed=42, report_to="none")

    class CommitCheckpoint(TrainerCallback):
        def on_save(self, args, state, control, **kwargs):
            artifacts.commit()
            print(f"Checkpoint persisted: {state.global_step}", flush=True)
            return control

    trainer = SFTTrainer(model=model, args=args, train_dataset=ds,
                         processing_class=processor, peft_config=lora)
    trainer.add_callback(CommitCheckpoint())
    Path(ROOT).mkdir(parents=True, exist_ok=True)
    config = {"model": MODEL_ID, "training_examples": len(ds), "steps": steps,
              "learning_rate": 3e-5, "seed": 42, "lora_rank": 16,
              "effective_batch_size": 8, "run_type": "independent_from_original_model"}
    Path(f"{ROOT}/config.json").write_text(json.dumps(config, indent=2))
    artifacts.commit()
    trainer.train()
    trainer.save_model(f"{ROOT}/adapter")
    processor.save_pretrained(f"{ROOT}/adapter")
    Path(f"{ROOT}/training_log.json").write_text(json.dumps(trainer.state.log_history, indent=2))
    artifacts.commit()
    print(f"Saved final adapter at {ROOT}/adapter", flush=True)


@app.function(image=image, gpu="L4", cpu=4, memory=16384, timeout=7200,
              volumes=VOLUMES, env=ENV)
def evaluate(count: int, checkpoint: str):
    from datasets import load_dataset
    checkpoint = str(checkpoint).strip()
    allowed = {"250", "500", "750", "1000", "1250", "final"}
    if checkpoint not in allowed:
        raise ValueError(f"Choose checkpoint from {sorted(allowed)}")
    path = (f"{ROOT}/adapter" if checkpoint == "final"
            else f"{ROOT}/checkpoints/checkpoint-{checkpoint}")
    if not Path(path).exists():
        raise FileNotFoundError(f"Adapter checkpoint missing: {path}")
    ds = load_dataset(DATASET_ID, split="val")
    count = min(count, len(ds))
    model, processor = load_model(path)
    rows = []
    for idx in range(count):
        item = ds[idx]
        pred = answer(model, processor, item)
        row = {"index": idx, "question": item["query"],
               "reference": item["label"], "prediction": pred,
               "human_or_machine": str(item["human_or_machine"]),
               "correct": score_chartqa(pred, item["label"])}
        rows.append(row)
        if (idx + 1) % 25 == 0 or idx + 1 == count:
            print(f"[{idx+1}/{count}] accuracy = {sum(x['correct'] for x in rows)/(idx+1):.2%}", flush=True)
    accuracy = sum(r["correct"] for r in rows) / len(rows)
    output = f"{ROOT}/eval_{checkpoint}_val_{count}.json"
    Path(output).write_text(json.dumps({"model": MODEL_ID, "checkpoint": checkpoint,
                                        "count": count, "relaxed_accuracy": accuracy,
                                        "results": rows}, indent=2))
    artifacts.commit()
    print(f"Checkpoint {checkpoint}: {accuracy:.2%} -> {output}", flush=True)
    return {"checkpoint": checkpoint, "accuracy": accuracy, "saved": output}


@app.function(image=modal.Image.debian_slim(python_version="3.11"), cpu=1,
              volumes={"/artifacts": artifacts}, timeout=300)
def compare(count: int = 200):
    baseline_path = Path(f"/artifacts/baseline_val_1920.json")
    if not baseline_path.exists():
        raise FileNotFoundError("Missing earlier baseline predictions on the existing Volume")
    base = json.loads(baseline_path.read_text())["results"]
    base_by_idx = {r["index"]: r for r in base}
    summary = {}
    for ck in ["250", "500", "750", "1000", "1250", "final"]:
        path = Path(f"{ROOT}/eval_{ck}_val_{count}.json")
        if not path.exists():
            continue
        rows = json.loads(path.read_text())["results"]
        wins = losses = year_error = scale_error = 0
        groups = {}
        for r in rows:
            b = base_by_idx[r["index"]]
            # Recalculate both using the SAME scorer, without trusting old flags.
            old = score_chartqa(b["prediction"], b["reference"])
            new = score_chartqa(r["prediction"], r["reference"])
            wins += (not old and new)
            losses += (old and not new)
            group = r["human_or_machine"]
            if group not in groups:
                groups[group] = [0, 0]
            groups[group][0] += 1
            groups[group][1] += int(new)
            q = r["question"].lower()
            if ("year" in q or "when" in q) and re.fullmatch(r"\d{4}", str(r["reference"][0]).strip()):
                if str(r["prediction"]).strip() != str(r["reference"][0]).strip():
                    year_error += 1
            scale_error += diagnostic_label(r["prediction"], r["reference"]) is not None
        summary[ck] = {
            "accuracy": sum(score_chartqa(r["prediction"], r["reference"]) for r in rows)/len(rows),
            "baseline_on_same_subset": sum(score_chartqa(base_by_idx[r["index"]]["prediction"],r["reference"]) for r in rows)/len(rows),
            "fixes": wins, "regressions": losses,
            "human_machine_accuracy": {g: round(c/n, 4) for g, (n,c) in groups.items()},
            "year_answer_mismatches_diagnostic": year_error,
            "possible_100x_scale_mismatches": scale_error}
    output = f"{ROOT}/comparison_val_{count}.json"
    Path(output).write_text(json.dumps(summary, indent=2))
    artifacts.commit()
    print(json.dumps(summary, indent=2))
    print("Comparison saved:", output)
    return summary


@app.local_entrypoint()
def main(stage: str = "compare", count: int = 200, steps: int = 1250,
         checkpoint: str = "final"):
    if stage == "train":
        return train.remote(count=count, steps=steps)
    if stage == "evaluate":
        return evaluate.remote(count=count, checkpoint=checkpoint)
    if stage == "compare":
        return compare.remote(count=count)
    raise ValueError("stage must be train, evaluate, or compare")
