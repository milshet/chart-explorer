#!/usr/bin/env python3
"""Evaluate the base model or a trained adapter on val or test with the Week 1 scorer (#19, #7, #8).

  # login shell, CPU only: check settings, data, prompts and that the scorer accepts every gold answer
  python scripts/evaluate.py --adapter ~/runs/run_a_r16/adapter --split val --dry-run

  # GPU, through Slurm
  sbatch slurm/eval.sbatch --adapter ~/runs/run_a_r16/adapter --split val              # final adapter
  sbatch slurm/eval.sbatch --adapter ~/runs/run_a_r16/checkpoints/epoch-1 --split val  # an epoch snapshot
  sbatch slurm/eval.sbatch --split test --targets answer_only --shots 2 \\
      --precision 4bit --max-new-tokens 384                                              # a Week 1 baseline

With --adapter, the variant, targets and image size come from the run's own config (adapter/train_config.json,
or <run>/config.json for checkpoints/epoch-N), so the prompts match training; passing a different value is an
error. Decoding is greedy. Prompts are built from the dataset's messages exactly as in training (the
assistant turn removed); few-shot examples are the two train conversations pinned in Week 1.

Writes to --out (default ~/evals/<name>): predictions.jsonl (one line per item: raw text, parsed JSON, score,
tokens, truncation, latency) and summary.json (metrics overall and per question type, provenance, settings).
Like train.py, a GPU run refuses to start unless the data matches MANIFEST.txt, the code is committed and the
packages match requirements-vm03.lock (--allow-unclean for throwaway tests, recorded as clean=false).
"""
import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import scoring  # noqa: E402  the Week 1 scorer
import smoke_test as st  # noqa: E402  MODEL_ID, MAX_PIXELS, load_image, model_revision
import train  # noqa: E402  provenance helpers: data fingerprint, git state, environment lock

# The two train examples behind the reported Week 1 few-shot baseline (notebooks/01_vlm_baselines.ipynb, cell 5).
FEW_SHOT_IDS = ["train-multi_col_20389-6093", "train-87453827004907-3340"]
STYLE = {"reasoning": "cot", "answer_only": "answer_only"}  # targets -> scorer style


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--adapter", type=Path, help="adapter/ or checkpoints/epoch-N of a run; omit for the base model")
    p.add_argument("--split", choices=("val", "test"), required=True)
    p.add_argument("--variant", choices=("image_only", "image_table"))
    p.add_argument("--targets", choices=tuple(STYLE), help="required for the base model; read from the run otherwise")
    p.add_argument("--shots", type=int, choices=(0, len(FEW_SHOT_IDS)), default=0)
    p.add_argument("--precision", choices=("bf16", "4bit"), default="bf16",
                   help="4bit = the Week 1 notebook's NF4 loading, to reproduce its baselines")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--limit", type=int, help="first N items only (smoke tests; not a reportable result)")
    p.add_argument("--final-test", action="store_true",
                   help="required to score an adapter on test: only after the final model is chosen on val, once per compared model (docs/RUNS.md rule 4)")
    p.add_argument("--out", type=Path)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--allow-unclean", action="store_true")
    return p.parse_args(argv)


def adapter_config(adapter):
    """The training config of the run an adapter or epoch checkpoint came from."""
    adapter = adapter.expanduser().resolve()
    if not (adapter / "adapter_model.safetensors").is_file():
        sys.exit(f"{adapter} has no adapter_model.safetensors")
    for cfg_file in (adapter / "train_config.json", adapter.parent.parent / "config.json"):
        if cfg_file.is_file():
            return adapter, json.loads(cfg_file.read_text(encoding="utf-8")), cfg_file
    sys.exit(f"no train_config.json in {adapter} and no config.json two levels up: cannot tell how it was trained")


def resolve(args):
    """Settings for this evaluation: {data_dir, variant, targets, style, source, ...}."""
    s = {"data_dir": "data/v2.0", "variant": args.variant or "image_only", "targets": args.targets,
         "adapter": None, "run_name": None, "source": "base"}
    if args.adapter:
        adapter, cfg, cfg_file = adapter_config(args.adapter)
        for key in ("variant", "targets"):
            given = getattr(args, key)
            if given and given != cfg[key]:
                sys.exit(f"--{key} {given} differs from the run's {key} {cfg[key]} ({cfg_file}); prompts must match training")
        if cfg["max_pixels"] != st.MAX_PIXELS:
            sys.exit(f"the run used max_pixels={cfg['max_pixels']}, but images are loaded at {st.MAX_PIXELS}")
        s.update(data_dir=cfg["data_dir"], variant=cfg["variant"], targets=cfg["targets"], adapter=str(adapter),
                 run_name=cfg["run_name"], source=f"{cfg['run_name']}-{adapter.name}")
        if args.split == "test" and not args.final_test:
            sys.exit("adapters are scored on test only after the final model is chosen on val, once each (docs/RUNS.md rule 4): add --final-test")
    elif not args.targets:
        sys.exit("--targets is required for the base model (answer_only for zero/few-shot, reasoning for CoT)")
    s["style"] = STYLE[s["targets"]]
    s["name"] = f"{s['source']}_{args.split}_{s['variant']}_{s['style']}_{args.shots}shot_{args.precision}" + \
        (f"_limit{args.limit}" if args.limit else "")
    return s


def load_rows(s, split, args):
    cfg = {k: s[k] for k in ("data_dir", "variant", "targets")}
    items = train.read_split(cfg, split)
    rows = {split: items}
    shots = []
    if args.shots:
        by_id = {r["id"]: r for r in train.read_split(cfg, "train")}
        missing = [i for i in FEW_SHOT_IDS if i not in by_id]
        if missing:
            sys.exit(f"pinned few-shot examples not in the train split: {missing}")
        shots = [by_id[i] for i in FEW_SHOT_IDS[:args.shots]]
        rows["train"] = shots
    fingerprint = train.data_fingerprint(cfg, rows)  # exits if the split files differ from MANIFEST.txt
    return (items[:args.limit] if args.limit else items), shots, fingerprint


def build(item, shots, data_dir):
    """(messages, images): the eval item's system and user turns, after any solved train conversations."""
    msgs, images = [item["messages"][0]], []
    for s in shots:
        msgs += s["messages"][1:3]
        images.append(st.load_image(data_dir, s["image"]))
    msgs.append(item["messages"][1])
    images.append(st.load_image(data_dir, item["image"]))
    return msgs, images


def prompt_text(processor, msgs):
    return processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def provenance(fingerprint, adapter):
    commit, dirty = train.git_state()
    pins, installed = train.read_lock(), train.installed_versions()
    drift = train.env_drift(pins, installed)
    problems = []
    if commit is None:
        problems.append("not a git checkout")
    if dirty:
        problems.append(f"uncommitted changes in {len(dirty)} tracked file(s): {', '.join(dirty[:5])}")
    if not pins:
        problems.append(f"{train.LOCK_FILE.name} is missing")
    elif drift:
        problems.append(f"{len(drift)} package(s) differ from {train.LOCK_FILE.name}: "
                        + ", ".join(f"{n} {a} -> {b}" for n, (a, b) in list(drift.items())[:5]))
    prov = {"git_commit": commit, "git_dirty": dirty,
            "versions": {"python": platform.python_version(), **{p: installed.get(p) for p in train.RECORDED_PACKAGES}},
            "env_drift": {n: {"locked": a, "installed": b} for n, (a, b) in drift.items()},
            "data": fingerprint, "clean": not problems}
    if adapter:
        a = Path(adapter)
        prov["adapter_sha256"] = train.sha256_file(a / "adapter_model.safetensors")
        run_prov = next((p for p in (a.parent / "provenance.json", a.parent.parent / "provenance.json") if p.is_file()), None)
        if run_prov:
            trained = json.loads(run_prov.read_text(encoding="utf-8"))
            prov["trained_at_commit"] = trained.get("git_commit")
            prov["trained_data"] = trained.get("data")
            if trained.get("data", {}).get("data_dir") != fingerprint["data_dir"]:
                problems.append(f"adapter was trained on {trained['data'].get('data_dir')}, evaluated on {fingerprint['data_dir']}")
                prov["clean"] = False
    return prov, problems


def dry_run(args, s, items, shots, prov, problems):
    """CPU checks: every prompt builds, and the scorer gives full marks to every gold answer."""
    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(st.MODEL_ID, local_files_only=True)
    data_dir = train.REPO / s["data_dir"]
    lengths, bad_gold = [], []
    for item in items:
        msgs, images = build(item, shots, data_dir)
        text = prompt_text(processor, msgs)
        lengths.append(int(processor(text=[text], images=images, return_tensors="pt")["input_ids"].shape[1]))
        gold = scoring.score(item, item["messages"][2]["content"], s["style"])
        if not (gold["correct"] and gold["schema_valid"]):
            bad_gold.append(item["id"])
    print(json.dumps({"settings": s, "items": len(items), "shots": [r["id"] for r in shots],
                      "prompt_tokens": {"max": max(lengths), "mean": round(sum(lengths) / len(lengths), 1)},
                      "provenance": prov}, indent=2))
    for p in problems:
        print(f"WARNING: a GPU run would refuse to start: {p}")
    if bad_gold:
        sys.exit(f"the scorer rejects {len(bad_gold)} gold answer(s), e.g. {bad_gold[:3]}")
    print("EVAL_DRY_RUN_OK (every prompt built, every gold answer scores correct and schema-valid)")


def load_model(torch, precision, adapter):
    from transformers import AutoModelForImageTextToText, BitsAndBytesConfig
    quant = None
    if precision == "4bit":  # exactly the Week 1 notebook's loading (cell 7)
        quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                                   bnb_4bit_compute_dtype=torch.bfloat16)
    model = AutoModelForImageTextToText.from_pretrained(st.MODEL_ID, dtype=torch.bfloat16, device_map={"": 0},
                                                        quantization_config=quant, local_files_only=True)
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter, local_files_only=True)
    return model.eval()


def stop_ids(model, tokenizer):
    eos = model.generation_config.eos_token_id
    ids = set(eos if isinstance(eos, (list, tuple)) else [eos])
    ids.add(tokenizer.pad_token_id)
    return {i for i in ids if i is not None}


def answer_length(tokens, stops):
    """(tokens generated before the first stop token, True if no stop token = hit max_new_tokens)."""
    for n, t in enumerate(tokens):
        if t in stops:
            return n, False
    return len(tokens), True


def main(argv=None):
    args = parse_args(argv)
    s = resolve(args)
    items, shots, fingerprint = load_rows(s, args.split, args)
    prov, problems = provenance(fingerprint, s["adapter"])
    if args.dry_run:
        return dry_run(args, s, items, shots, prov, problems)
    if not os.getenv("SLURM_JOB_ID"):
        sys.exit("Submit with sbatch (slurm/eval.sbatch); the login shell has no GPU. --dry-run works here.")
    if problems and not args.allow_unclean:
        sys.exit("Not starting: this evaluation would not be reproducible:\n  " + "\n  ".join(problems)
                 + "\nCommit or revert the changes / restore the locked environment, or pass --allow-unclean.")

    out = (args.out or Path.home() / "evals" / s["name"]).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=False)  # never overwrite an earlier evaluation
    print(f"evaluating {s['source']} on {args.split} ({len(items)} items) -> {out}", flush=True)

    import torch
    from transformers import AutoProcessor
    t_start = time.monotonic()
    processor = AutoProcessor.from_pretrained(st.MODEL_ID, local_files_only=True)
    processor.tokenizer.padding_side = "left"  # batched generation continues from the right edge
    model = load_model(torch, args.precision, s["adapter"])
    stops = stop_ids(model, processor.tokenizer)
    load_seconds = time.monotonic() - t_start
    data_dir = train.REPO / s["data_dir"]

    results, extra = [], {}
    gen_seconds = 0.0
    with open(out / "predictions.jsonl", "w", encoding="utf-8") as pf:
        for i in range(0, len(items), args.batch_size):
            chunk = items[i:i + args.batch_size]
            built = [build(it, shots, data_dir) for it in chunk]
            batch = processor(text=[prompt_text(processor, m) for m, _ in built], images=[im for _, im in built],
                              padding=True, return_tensors="pt").to(model.device)
            torch.cuda.synchronize()
            t0 = time.monotonic()
            with torch.no_grad():
                gen = model.generate(**batch, max_new_tokens=args.max_new_tokens, do_sample=False,
                                     pad_token_id=processor.tokenizer.pad_token_id)
            torch.cuda.synchronize()
            secs = time.monotonic() - t0
            gen_seconds += secs
            new = gen[:, batch["input_ids"].shape[1]:]
            for item, seq, raw in zip(chunk, new.tolist(), processor.batch_decode(new, skip_special_tokens=True)):
                res = scoring.score(item, raw, s["style"])
                n_tok, truncated = answer_length(seq, stops)
                extra[item["id"]] = {"new_tokens": n_tok, "truncated": truncated}
                results.append(res)
                gold = json.loads(item["messages"][2]["content"])
                pf.write(json.dumps({**res, "parsed": scoring.extract_json(raw), "gold": gold.get("answer"),
                                     "question_type": item["question_type"], "answer_type": item["answer_type"],
                                     "new_tokens": n_tok, "truncated": truncated,
                                     "seconds": round(secs / len(chunk), 3)}, ensure_ascii=False) + "\n")
            pf.flush()
            done = min(i + args.batch_size, len(items))
            print(f"  {done}/{len(items)}  acc so far {sum(r['correct'] for r in results) / len(results):.3f}", flush=True)

    overall, by = scoring.summarise(results, items)
    tokens = sum(e["new_tokens"] for e in extra.values())
    overall.update({
        "truncated": sum(e["truncated"] for e in extra.values()),
        "mean_new_tokens": round(tokens / len(items), 1),
        "tokens_per_second": round(tokens / gen_seconds, 1) if gen_seconds else None,
        "seconds_per_item": round(gen_seconds / len(items), 3),
    })
    summary = {
        "name": s["name"], "settings": {**s, "split": args.split, "shots": args.shots, "shot_ids": [r["id"] for r in shots],
                                        "precision": args.precision, "batch_size": args.batch_size,
                                        "max_new_tokens": args.max_new_tokens, "limit": args.limit, "decoding": "greedy"},
        "model": st.MODEL_ID, "model_revision": st.model_revision(), "gpu": torch.cuda.get_device_name(0),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "provenance": prov,
        "reportable": prov["clean"] and not args.limit,
        "overall": {k: (round(v, 4) if isinstance(v, float) else v) for k, v in overall.items()},
        "by": by, "load_seconds": round(load_seconds, 1), "minutes": round((time.monotonic() - t_start) / 60, 1),
        "peak_gpu_memory_mib": round(torch.cuda.max_memory_allocated() / 2**20),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary["overall"], indent=2))
    print("by question_type:", json.dumps(by["question_type"]))
    print(f"EVAL_DONE {out}")


if __name__ == "__main__":
    main()
