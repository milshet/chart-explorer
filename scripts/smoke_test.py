#!/usr/bin/env python3
"""Smoke test for the VM03 training setup: Qwen2.5-VL-3B with QLoRA on chart questions.

Runs, inside one short Slurm GPU job, every stage the Week 2 runs depend on:

  1. load    Qwen2.5-VL-3B in 4-bit on the GPU
  2. infer   answer a few test charts (zero-shot) and save the raw outputs
  3. train   a few QLoRA steps on training examples, loss on the answer only
  4. learn   loss on a trained example must drop after training
  5. reload  save the adapter, load it again on a fresh base model, answer one chart

It writes summary.json (pass/fail per check, peak GPU memory, seconds per step),
raw_outputs.jsonl and losses.jsonl to --out, prints SMOKE_PASS or SMOKE_FAIL,
and exits non-zero on failure. Submit with slurm/smoke.sbatch.
"""
import argparse
import gc
import json
import math
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
MAX_PIXELS = 800 * 800  # same limit as notebooks/01_vlm_baselines.ipynb
LORA_TARGETS = r"^(?!.*visual).*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)$"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", type=Path, default=Path("data/v1.0"))
    p.add_argument("--variant", default="image_only")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--n-infer", type=int, default=5)
    p.add_argument("--n-train", type=int, default=20)
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--rank", type=int, default=16)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def guard(args):
    """Same rules as upGrad's starter: GPU only via Slurm, output only inside your own home."""
    if not os.getenv("SLURM_JOB_ID"):
        sys.exit("Submit with sbatch (slurm/smoke.sbatch); the login shell has no GPU.")
    home = Path.home().resolve()
    out = args.out.expanduser().resolve()
    if home not in out.parents:
        sys.exit(f"--out must be inside {home}")
    out.mkdir(parents=True, exist_ok=False)  # never overwrite an earlier run
    args.out = out


def read_jsonl(path, n):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
            if len(rows) == n:
                break
    return rows


def load_image(data_dir, rel):
    from PIL import Image
    img = Image.open(data_dir / rel).convert("RGB")
    w, h = img.size
    if w * h > MAX_PIXELS:
        s = (MAX_PIXELS / (w * h)) ** 0.5
        img = img.resize((max(28, int(w * s)), max(28, int(h * s))), Image.BICUBIC)
    return img


def git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return None


def model_revision():
    from huggingface_hub import snapshot_download
    return Path(snapshot_download(MODEL_ID, local_files_only=True)).name


def load_base(torch):
    from transformers import BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration
    quant = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        llm_int8_skip_modules=["visual", "lm_head"],  # keep the vision encoder and output layer in bf16
    )
    return Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_ID, quantization_config=quant, dtype=torch.bfloat16, device_map={"": 0},
        local_files_only=True,
    )


def encode(processor, messages, image, prompt_only):
    msgs = messages[:-1] if prompt_only else messages
    text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=prompt_only)
    return processor(text=[text], images=[image], return_tensors="pt")


def training_batch(processor, row, image):
    """Tokenise one example and mask everything except the assistant's answer."""
    prompt = encode(processor, row["messages"], image, prompt_only=True)
    full = encode(processor, row["messages"], image, prompt_only=False)
    n = prompt["input_ids"].shape[1]
    if not bool((full["input_ids"][0, :n] == prompt["input_ids"][0]).all()):
        raise RuntimeError(f"{row['id']}: prompt tokens are not a prefix of the full example; masking would be wrong")
    labels = full["input_ids"].clone()
    labels[0, :n] = -100
    full["labels"] = labels
    return full, int((labels != -100).sum()), int(full["input_ids"].shape[1])


def generate(torch, model, processor, row, image, max_new_tokens):
    batch = encode(processor, row["messages"], image, prompt_only=True).to(model.device)
    t0 = time.monotonic()
    with torch.no_grad():
        out = model.generate(**batch, max_new_tokens=max_new_tokens, do_sample=False)
    new = out[0, batch["input_ids"].shape[1]:]
    text = processor.decode(new, skip_special_tokens=True).strip()
    return text, int(new.shape[0]), time.monotonic() - t0


def is_valid_json(text):
    s = text.strip()
    if s.startswith("```"):
        s = s.strip("`").removeprefix("json").strip()
    try:
        return isinstance(json.loads(s), dict)
    except ValueError:
        return False


def main():
    args = parse_args()
    guard(args)

    import torch
    import transformers
    import peft
    from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoProcessor

    if not torch.cuda.is_available():
        sys.exit("The Slurm job has no usable GPU")
    torch.manual_seed(args.seed)

    checks, notes = {}, []
    summary = {
        "model": MODEL_ID, "model_revision": model_revision(), "git_commit": git_commit(),
        "slurm_job_id": os.environ["SLURM_JOB_ID"], "gpu": torch.cuda.get_device_name(0),
        "versions": {"python": platform.python_version(), "torch": torch.__version__,
                     "transformers": transformers.__version__, "peft": peft.__version__},
        "settings": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "max_pixels": MAX_PIXELS, "checks": checks, "notes": notes,
    }
    data_dir = args.data_dir.resolve()
    test_rows = read_jsonl(data_dir / args.variant / "test.jsonl", args.n_infer)
    train_rows = read_jsonl(data_dir / args.variant / "train.jsonl", args.n_train)
    raw_out = open(args.out / "raw_outputs.jsonl", "w", encoding="utf-8")

    def finish():
        raw_out.close()
        summary["peak_gpu_memory_mib"] = round(torch.cuda.max_memory_allocated() / 2**20)
        summary["passed"] = bool(checks) and all(checks.values())
        (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(json.dumps(summary, indent=2), flush=True)
        print("SMOKE_PASS" if summary["passed"] else "SMOKE_FAIL", flush=True)
        sys.exit(0 if summary["passed"] else 1)

    # 1. load ------------------------------------------------------------------
    t0 = time.monotonic()
    processor = AutoProcessor.from_pretrained(MODEL_ID, local_files_only=True)
    model = load_base(torch)
    model.eval()
    summary["load_seconds"] = round(time.monotonic() - t0, 1)
    checks["1_model_loads_on_gpu"] = next(model.parameters()).is_cuda
    print(f"[1] model loaded in {summary['load_seconds']}s", flush=True)

    # 2. zero-shot inference ----------------------------------------------------
    valid, gen_tokens, gen_secs = 0, 0, 0.0
    for row in test_rows:
        text, n_tok, secs = generate(torch, model, processor, row, load_image(data_dir, row["image"]), args.max_new_tokens)
        ok = is_valid_json(text)
        valid += ok
        gen_tokens += n_tok
        gen_secs += secs
        raw_out.write(json.dumps({"stage": "zero_shot", "id": row["id"], "gold": row["answer"],
                                  "output": text, "valid_json": ok, "new_tokens": n_tok}) + "\n")
    checks["2_inference_produces_text"] = gen_tokens > 0
    summary["zero_shot"] = {"items": len(test_rows), "valid_json": valid,
                            "tokens_per_second": round(gen_tokens / gen_secs, 1) if gen_secs else None}
    print(f"[2] zero-shot: {valid}/{len(test_rows)} valid JSON, {summary['zero_shot']['tokens_per_second']} tok/s", flush=True)

    # 3. QLoRA training ------------------------------------------------------------
    batches = []
    for row in train_rows:
        b, n_answer, n_total = training_batch(processor, row, load_image(data_dir, row["image"]))
        batches.append(b)
    checks["3a_loss_masks_prompt"] = True  # training_batch raises if masking is wrong
    summary["train_tokens"] = {"max_sequence": max(int(b["input_ids"].shape[1]) for b in batches),
                               "mean_answer_tokens": round(sum(int((b["labels"] != -100).sum()) for b in batches) / len(batches), 1)}

    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True,
                                            gradient_checkpointing_kwargs={"use_reentrant": False})
    model.config.use_cache = False
    model = get_peft_model(model, LoraConfig(r=args.rank, lora_alpha=2 * args.rank, lora_dropout=0.05,
                                             target_modules=LORA_TARGETS, task_type="CAUSAL_LM"))
    lora_modules = [n for n, m in model.named_modules() if hasattr(m, "lora_A") and len(getattr(m, "lora_A", {})) > 0]
    checks["3b_vision_encoder_frozen"] = len(lora_modules) > 0 and not any("visual" in n for n in lora_modules)
    trainable = [p for p in model.parameters() if p.requires_grad]
    summary["lora"] = {"wrapped_modules": len(lora_modules), "trainable_parameters": sum(p.numel() for p in trainable)}

    def loss_of(b):
        with torch.no_grad():
            return float(model(**{k: v.to(model.device) for k, v in b.items()}).loss)

    model.eval()
    probe_before = loss_of(batches[0])
    probe_param = next(p for n, p in model.named_parameters() if "lora_B" in n)
    weights_before = probe_param.detach().float().cpu().clone()

    model.train()
    optimizer = torch.optim.AdamW(trainable, lr=args.lr)
    losses, step_secs = [], []
    with open(args.out / "losses.jsonl", "w", encoding="utf-8") as lf:
        for step in range(1, args.steps + 1):
            b = {k: v.to(model.device) for k, v in batches[(step - 1) % len(batches)].items()}
            torch.cuda.synchronize()
            t = time.monotonic()
            optimizer.zero_grad(set_to_none=True)
            loss = model(**b).loss
            if not torch.isfinite(loss):
                notes.append(f"non-finite loss at step {step}")
                break
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            torch.cuda.synchronize()
            step_secs.append(time.monotonic() - t)
            losses.append(loss.item())
            lf.write(json.dumps({"step": step, "loss": round(losses[-1], 5), "seconds": round(step_secs[-1], 3)}) + "\n")
            print(f"    step {step:3d}  loss {losses[-1]:.4f}  {step_secs[-1]:.2f}s", flush=True)

    checks["3c_training_losses_finite"] = len(losses) == args.steps and all(math.isfinite(x) for x in losses)
    checks["3d_lora_weights_changed"] = not torch.equal(weights_before, probe_param.detach().float().cpu())
    later = step_secs[1:] or step_secs  # step 1 includes warm-up
    summary["training"] = {"steps": len(losses), "first_loss": losses[0] if losses else None,
                           "last_loss": losses[-1] if losses else None,
                           "seconds_per_step": round(sum(later) / len(later), 3) if later else None}

    # 4. did it learn? -----------------------------------------------------------
    model.eval()
    probe_after = loss_of(batches[0])
    summary["learning_probe"] = {"example": train_rows[0]["id"], "loss_before": round(probe_before, 4),
                                 "loss_after": round(probe_after, 4)}
    checks["4_loss_drops_on_trained_example"] = probe_after < probe_before
    print(f"[3-4] trained {len(losses)} steps, {summary['training']['seconds_per_step']}s/step; "
          f"probe loss {probe_before:.3f} -> {probe_after:.3f}", flush=True)

    # 5. save, reload on a fresh base model, answer one chart -----------------------
    adapter_dir = args.out / "adapter"
    model.save_pretrained(adapter_dir, safe_serialization=True)
    processor.save_pretrained(adapter_dir)
    del model, optimizer, batches
    gc.collect()
    torch.cuda.empty_cache()

    reloaded = PeftModel.from_pretrained(load_base(torch), adapter_dir, local_files_only=True)
    reloaded.eval()
    row = test_rows[0]
    text, n_tok, _ = generate(torch, reloaded, processor, row, load_image(data_dir, row["image"]), args.max_new_tokens)
    raw_out.write(json.dumps({"stage": "after_reload", "id": row["id"], "gold": row["answer"],
                              "output": text, "valid_json": is_valid_json(text), "new_tokens": n_tok}) + "\n")
    checks["5_adapter_saves_and_reloads"] = (adapter_dir / "adapter_model.safetensors").is_file() and n_tok > 0
    summary["adapter_mib"] = round((adapter_dir / "adapter_model.safetensors").stat().st_size / 2**20, 1)
    print(f"[5] adapter reloaded ({summary['adapter_mib']} MiB), generated {n_tok} tokens", flush=True)

    finish()


if __name__ == "__main__":
    main()
