#!/usr/bin/env python3
"""QLoRA fine-tuning of Qwen2.5-VL-3B on ChartAudit-Fin (Week 2).

One YAML file in configs/ describes one run. The base model is loaded in 4-bit, LoRA is applied to
the language model only (vision encoder frozen), and the loss covers the assistant's answer only.
Model loading, chat encoding and loss masking come from scripts/smoke_test.py, so training uses the
same code the smoke test checks.

  # login shell, CPU only: check the config and tokenise every train/val example
  python scripts/train.py configs/run_a.yaml --dry-run

  # GPU, through Slurm
  sbatch slurm/train.sbatch configs/run_a.yaml                                       # full run
  sbatch slurm/train.sbatch configs/run_a.yaml --max-steps 50 --out ~/runs/pilot     # pilot
  sbatch slurm/train.sbatch configs/run_a.yaml --resume                              # continue a run
  sbatch slurm/train.sbatch configs/run_a.yaml --set train.lr=1e-4 --set run_name=run_a_lr1e-4

Writes to --out (default ~/runs/<run_name>): metrics.jsonl (one line per step and per validation),
checkpoints/step-NNNNN (resumable, newest `save.keep` kept), adapter/ (final adapter, processor and
train_config.json) and run_summary.json. Logs to Weights & Biases when `wandb.enabled` is true and
the wandb package is installed; otherwise metrics.jsonl is the record.
"""
import argparse
import copy
import json
import math
import os
import platform
import random
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import smoke_test as st  # noqa: E402  shared helpers: load_base, encode, training_batch, load_image

REPO = Path(__file__).resolve().parents[1]

DEFAULTS = {
    "run_name": None,
    "data_dir": "data/v2.0",
    "variant": "image_only",          # image_only | image_table
    "targets": "reasoning",           # reasoning (evidence + rationale + answer) | answer_only
    "max_pixels": 800 * 800,          # same limit as notebooks/01_vlm_baselines.ipynb
    "max_seq_len": 2048,              # longer examples are skipped and counted
    "lora": {"r": 16, "alpha": 32, "dropout": 0.05, "target_modules": st.LORA_TARGETS},
    "train": {"epochs": 2, "lr": 2.0e-4, "grad_accum": 8, "warmup_ratio": 0.03,
              "weight_decay": 0.0, "max_grad_norm": 1.0, "seed": 42},
    "eval": {"every_steps": 50},      # validation loss every N optimizer steps (0 = end of epoch only)
    "save": {"every_steps": 50, "keep": 2},
    "wandb": {"enabled": True, "project": "chartqa-week2", "entity": None},
}


# --- config --------------------------------------------------------------------------------------

def load_config(path, overrides):
    import yaml
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    for item in overrides:
        key, sep, value = item.partition("=")
        if not sep:
            sys.exit(f"--set expects key=value, got '{item}'")
        node = raw
        *parents, leaf = key.split(".")
        for p in parents:
            node = node.setdefault(p, {})
        node[leaf] = yaml.safe_load(value)

    cfg = copy.deepcopy(DEFAULTS)

    def merge(dst, src, prefix=""):
        for k, v in src.items():
            if k not in dst:
                sys.exit(f"{path}: unknown setting '{prefix}{k}'")
            if isinstance(dst[k], dict):
                if not isinstance(v, dict):
                    sys.exit(f"{path}: '{prefix}{k}' must be a mapping")
                merge(dst[k], v, f"{prefix}{k}.")
            else:
                dst[k] = v

    merge(cfg, raw)
    if not cfg["run_name"]:
        sys.exit(f"{path}: run_name is required")
    if cfg["targets"] not in ("reasoning", "answer_only"):
        sys.exit(f"{path}: targets must be 'reasoning' or 'answer_only'")
    if cfg["train"]["grad_accum"] < 1 or cfg["train"]["epochs"] < 1:
        sys.exit(f"{path}: train.grad_accum and train.epochs must be at least 1")
    return cfg


def split_path(cfg, split):
    sub = "answer_only/" if cfg["targets"] == "answer_only" else ""
    return REPO / cfg["data_dir"] / cfg["variant"] / f"{sub}{split}.jsonl"


def read_split(cfg, split):
    path = split_path(cfg, split)
    if not path.is_file():
        sys.exit(f"missing {path}")
    return st.read_jsonl(path, None)


# --- data ----------------------------------------------------------------------------------------

def make_batch(processor, cfg, row):
    """Tokenised example with the prompt masked, or None if longer than max_seq_len."""
    b, n_answer, n_total = st.training_batch(processor, row, st.load_image(REPO / cfg["data_dir"], row["image"]))
    if n_total > cfg["max_seq_len"]:
        return None, n_answer, n_total
    return b, n_answer, n_total


def dry_run(cfg):
    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(st.MODEL_ID, local_files_only=True)
    report = {}
    for split in ("train", "val"):
        rows = read_split(cfg, split)
        lengths, answers, too_long = [], [], 0
        t0 = time.monotonic()
        for row in rows:
            b, n_answer, n_total = make_batch(processor, cfg, row)
            lengths.append(n_total)
            answers.append(n_answer)
            too_long += b is None
        report[split] = {"examples": len(rows), "too_long": too_long,
                         "max_seq_len_seen": max(lengths), "mean_seq_len": round(sum(lengths) / len(lengths), 1),
                         "mean_answer_tokens": round(sum(answers) / len(answers), 1),
                         "seconds": round(time.monotonic() - t0, 1)}
    n_train = report["train"]["examples"] - report["train"]["too_long"]
    steps = math.ceil(report["train"]["examples"] / cfg["train"]["grad_accum"]) * cfg["train"]["epochs"]
    print(json.dumps({"config": cfg, "data": report, "trainable_examples": n_train,
                      "optimizer_steps": steps}, indent=2))
    print("DRY_RUN_OK (every example tokenised and the prompt mask checked)")


# --- checkpoints ---------------------------------------------------------------------------------

def checkpoints(out):
    d = out / "checkpoints"
    return sorted(p for p in d.glob("step-*") if p.is_dir() and not p.name.endswith(".tmp")) if d.is_dir() else []


def save_checkpoint(torch, out, model, optimizer, scheduler, state, keep):
    final = out / "checkpoints" / f"step-{state['step']:05d}"
    tmp = final.with_name(final.name + ".tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    model.save_pretrained(tmp, safe_serialization=True)
    torch.save({"optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "state": state,
                "rng": {"python": random.getstate(), "torch": torch.get_rng_state(),
                        "cuda": torch.cuda.get_rng_state_all()}}, tmp / "trainer_state.pt")
    shutil.rmtree(final, ignore_errors=True)
    tmp.rename(final)  # a checkpoint only appears once it is complete
    for old in checkpoints(out)[:-keep]:
        shutil.rmtree(old, ignore_errors=True)
    print(f"    saved {final}", flush=True)


def load_checkpoint(torch, ckpt, model, optimizer, scheduler):
    from peft.utils import set_peft_model_state_dict
    from safetensors.torch import load_file
    result = set_peft_model_state_dict(model, load_file(ckpt / "adapter_model.safetensors"))
    if getattr(result, "unexpected_keys", None):
        sys.exit(f"{ckpt}: adapter does not match this config ({len(result.unexpected_keys)} unexpected keys)")
    blob = torch.load(ckpt / "trainer_state.pt", map_location="cpu", weights_only=False)
    optimizer.load_state_dict(blob["optimizer"])
    scheduler.load_state_dict(blob["scheduler"])
    random.setstate(blob["rng"]["python"])
    torch.set_rng_state(blob["rng"]["torch"])
    torch.cuda.set_rng_state_all(blob["rng"]["cuda"])
    return blob["state"]


# --- main ----------------------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("config", type=Path)
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                   help="override a config value, e.g. --set train.lr=1e-4 (repeatable)")
    p.add_argument("--out", type=Path, help="output directory (default ~/runs/<run_name>)")
    p.add_argument("--resume", action="store_true", help="continue from the newest checkpoint in --out")
    p.add_argument("--max-steps", type=int, help="stop after this many optimizer steps (pilot runs)")
    p.add_argument("--max-minutes", type=float, help="save a checkpoint and stop after this long (job time limit)")
    p.add_argument("--dry-run", action="store_true", help="check config and data on the CPU, no training")
    p.add_argument("--limit-train", type=int, help="use only the first N training examples (testing the script)")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config, args.set)
    st.MAX_PIXELS = cfg["max_pixels"]  # st.load_image reads this
    if args.dry_run:
        return dry_run(cfg)

    started = time.monotonic()
    if not os.getenv("SLURM_JOB_ID"):
        sys.exit("Submit with sbatch (slurm/train.sbatch); the login shell has no GPU. Use --dry-run to check data.")
    home = Path.home().resolve()
    out = (args.out or Path("~/runs") / cfg["run_name"]).expanduser().resolve()
    if home not in out.parents:
        sys.exit(f"--out must be inside {home}")
    if out.exists() and not args.resume:
        sys.exit(f"{out} already exists: pass --resume to continue it, or choose another --out / run_name")
    if args.resume and (out / "adapter" / "adapter_model.safetensors").is_file():
        sys.exit(f"{out} has already finished (adapter/ exists)")
    out.mkdir(parents=True, exist_ok=True)

    import torch
    import transformers
    import peft
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoProcessor, get_cosine_schedule_with_warmup

    if not torch.cuda.is_available():
        sys.exit("The Slurm job has no usable GPU")
    tc = cfg["train"]
    random.seed(tc["seed"])
    torch.manual_seed(tc["seed"])

    train_rows, val_rows = read_split(cfg, "train"), read_split(cfg, "val")
    if args.limit_train:
        train_rows = train_rows[:args.limit_train]
    steps_per_epoch = math.ceil(len(train_rows) / tc["grad_accum"])
    planned_steps = steps_per_epoch * tc["epochs"]
    stop_at = min(planned_steps, args.max_steps) if args.max_steps else planned_steps

    # model ----------------------------------------------------------------------------------------
    processor = AutoProcessor.from_pretrained(st.MODEL_ID, local_files_only=True)
    model = st.load_base(torch)
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True,
                                            gradient_checkpointing_kwargs={"use_reentrant": False})
    model.config.use_cache = False
    lc = cfg["lora"]
    model = get_peft_model(model, LoraConfig(r=lc["r"], lora_alpha=lc["alpha"], lora_dropout=lc["dropout"],
                                             target_modules=lc["target_modules"], task_type="CAUSAL_LM"))
    lora_modules = [n for n, m in model.named_modules() if hasattr(m, "lora_A") and len(getattr(m, "lora_A", {})) > 0]
    if not lora_modules or any("visual" in n for n in lora_modules):
        sys.exit("LoRA must wrap language-model layers only (vision encoder frozen)")
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=tc["lr"], weight_decay=tc["weight_decay"])
    scheduler = get_cosine_schedule_with_warmup(optimizer, max(1, round(tc["warmup_ratio"] * planned_steps)),
                                                planned_steps)

    state = {"step": 0, "epoch": 0, "next_index": 0, "skipped_too_long": 0, "val_history": [],
             "slurm_jobs": []}
    if args.resume:
        ckpts = checkpoints(out)
        if not ckpts:
            sys.exit(f"--resume: no checkpoint in {out / 'checkpoints'}")
        state = load_checkpoint(torch, ckpts[-1], model, optimizer, scheduler)
        print(f"resumed from {ckpts[-1]} (step {state['step']}, epoch {state['epoch'] + 1})", flush=True)
    state["slurm_jobs"].append(os.environ["SLURM_JOB_ID"])

    summary = {
        "run_name": cfg["run_name"], "status": "running", "model": st.MODEL_ID,
        "model_revision": st.model_revision(), "git_commit": st.git_commit(),
        "gpu": torch.cuda.get_device_name(0),
        "versions": {"python": platform.python_version(), "torch": torch.__version__,
                     "transformers": transformers.__version__, "peft": peft.__version__},
        "config": cfg, "data": {"train_file": str(split_path(cfg, "train").relative_to(REPO)),
                                "train_examples": len(train_rows), "val_examples": len(val_rows),
                                "limit_train": args.limit_train},
        "lora": {"wrapped_modules": len(lora_modules), "trainable_parameters": sum(p.numel() for p in trainable)},
        "planned_steps": planned_steps, "steps_per_epoch": steps_per_epoch,
    }
    (out / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")
    print(json.dumps({k: summary[k] for k in ("run_name", "data", "lora", "planned_steps")}, indent=2), flush=True)

    # logging --------------------------------------------------------------------------------------
    metrics = open(out / "metrics.jsonl", "a", encoding="utf-8")
    wandb_run = None
    if cfg["wandb"]["enabled"]:
        try:
            import wandb
            id_file = out / "wandb_id.txt"
            run_id = id_file.read_text().strip() if id_file.is_file() else wandb.util.generate_id()
            id_file.write_text(run_id)
            wandb_run = wandb.init(project=cfg["wandb"]["project"], entity=cfg["wandb"]["entity"],
                                   name=cfg["run_name"], id=run_id, resume="allow", config=cfg, dir=str(out))
            summary["wandb_url"] = getattr(wandb_run, "url", None)
        except Exception as e:  # never lose a GPU slot over logging
            print(f"W&B disabled ({type(e).__name__}: {e}); metrics.jsonl is the record", flush=True)

    def log(record):
        metrics.write(json.dumps(record) + "\n")
        metrics.flush()
        if wandb_run:
            wandb_run.log({k: v for k, v in record.items() if k not in ("kind", "step")}, step=record["step"])

    val_batches = []
    for row in val_rows:
        b, _, _ = make_batch(processor, cfg, row)
        if b is not None:
            val_batches.append(b)

    def validate(reason):
        model.eval()
        total = 0.0
        with torch.no_grad():
            for b in val_batches:
                total += float(model(**{k: v.to(model.device) for k, v in b.items()}).loss)
        model.train()
        val_loss = total / len(val_batches)
        state["val_history"].append({"step": state["step"], "val_loss": round(val_loss, 5), "at": reason})
        log({"kind": "val", "step": state["step"], "val/loss": val_loss})
        print(f"    val loss {val_loss:.4f} ({len(val_batches)} examples, {reason})", flush=True)

    def finish(status, code=0):
        summary["status"] = status
        summary["steps_done"] = state["step"]
        summary["skipped_too_long"] = state["skipped_too_long"]
        summary["val_history"] = state["val_history"]
        summary["slurm_jobs"] = state["slurm_jobs"]
        summary["peak_gpu_memory_mib"] = round(torch.cuda.max_memory_allocated() / 2**20)
        summary["minutes_this_job"] = round((time.monotonic() - started) / 60, 1)
        if step_secs:
            later = step_secs[1:] or step_secs  # the first step includes warm-up
            sps = sum(later) / len(later)
            summary["seconds_per_step"] = round(sps, 2)
            summary["estimated_train_minutes_full_run"] = round(sps * planned_steps / 60, 1)
        (out / "run_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(json.dumps({k: v for k, v in summary.items() if k != "config"}, indent=2), flush=True)
        if wandb_run:
            wandb_run.summary.update({k: summary[k] for k in ("status", "steps_done", "peak_gpu_memory_mib")})
            wandb_run.finish()
        metrics.close()
        print(f"TRAIN_{status.upper()}", flush=True)
        sys.exit(code)

    # training -------------------------------------------------------------------------------------
    step_secs = []
    deadline = started + args.max_minutes * 60 if args.max_minutes else None
    model.train()
    for epoch in range(state["epoch"], tc["epochs"]):
        order = list(range(len(train_rows)))
        random.Random(tc["seed"] + epoch).shuffle(order)  # same order on resume
        i = state["next_index"] if epoch == state["epoch"] else 0
        while i < len(order):
            chunk = order[i:i + tc["grad_accum"]]
            torch.cuda.synchronize()
            t0 = time.monotonic()
            optimizer.zero_grad(set_to_none=True)
            losses, answer_tokens = [], 0
            for idx in chunk:
                b, n_answer, _ = make_batch(processor, cfg, train_rows[idx])
                if b is None:
                    state["skipped_too_long"] += 1
                    continue
                loss = model(**{k: v.to(model.device) for k, v in b.items()}).loss
                if not torch.isfinite(loss):
                    save_checkpoint(torch, out, model, optimizer, scheduler, state, cfg["save"]["keep"])
                    finish(f"failed_nonfinite_loss_at_step_{state['step'] + 1}", 1)
                (loss / len(chunk)).backward()
                losses.append(loss.item())
                answer_tokens += n_answer
            i += len(chunk)
            state.update(epoch=epoch, next_index=i)
            if not losses:
                continue
            grad_norm = float(torch.nn.utils.clip_grad_norm_(trainable, tc["max_grad_norm"]))
            optimizer.step()
            scheduler.step()
            torch.cuda.synchronize()
            step_secs.append(time.monotonic() - t0)
            state["step"] += 1
            train_loss = sum(losses) / len(losses)
            log({"kind": "train", "step": state["step"], "epoch": round(epoch + i / len(order), 3),
                 "train/loss": train_loss, "train/lr": scheduler.get_last_lr()[0], "train/grad_norm": grad_norm,
                 "train/seconds_per_step": step_secs[-1], "train/answer_tokens": answer_tokens})
            print(f"step {state['step']:4d}/{stop_at}  epoch {epoch + 1}  loss {train_loss:.4f}  "
                  f"lr {scheduler.get_last_lr()[0]:.2e}  grad {grad_norm:.2f}  {step_secs[-1]:.1f}s", flush=True)

            if cfg["eval"]["every_steps"] and state["step"] % cfg["eval"]["every_steps"] == 0:
                validate(f"step {state['step']}")
            if cfg["save"]["every_steps"] and state["step"] % cfg["save"]["every_steps"] == 0:
                save_checkpoint(torch, out, model, optimizer, scheduler, state, cfg["save"]["keep"])
            if state["step"] >= stop_at and stop_at < planned_steps:
                validate("max_steps")
                save_checkpoint(torch, out, model, optimizer, scheduler, state, cfg["save"]["keep"])
                finish("stopped_at_max_steps")
            if deadline and time.monotonic() > deadline:
                save_checkpoint(torch, out, model, optimizer, scheduler, state, cfg["save"]["keep"])
                print("Time budget reached; resubmit with --resume to continue.", flush=True)
                finish("paused_time_limit", 75)

        state.update(epoch=epoch + 1, next_index=0)
        validate(f"end of epoch {epoch + 1}")
        save_checkpoint(torch, out, model, optimizer, scheduler, state, cfg["save"]["keep"])

    # final adapter --------------------------------------------------------------------------------
    adapter_dir = out / "adapter"
    model.save_pretrained(adapter_dir, safe_serialization=True)
    processor.save_pretrained(adapter_dir)
    (adapter_dir / "train_config.json").write_text(json.dumps(cfg, indent=2) + "\n")
    summary["adapter_dir"] = str(adapter_dir)
    summary["adapter_mib"] = round((adapter_dir / "adapter_model.safetensors").stat().st_size / 2**20, 1)
    if state["val_history"]:
        summary["final_val_loss"] = state["val_history"][-1]["val_loss"]
        summary["best_val_loss"] = min(v["val_loss"] for v in state["val_history"])
    finish("completed")


if __name__ == "__main__":
    main()
