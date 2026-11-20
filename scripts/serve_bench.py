#!/usr/bin/env python3
"""Served latency and tokens/sec (tasks 3.8, 3.14) on the serving stack decision 12 settled on: the adapter merged
into bf16 base weights, transformers.generate, batch 1, one request at a time.

  python scripts/serve_bench.py --adapter ~/runs/run_a_r16/adapter --dry-run          # login shell, no GPU
  sbatch slurm/serve_bench.sbatch --adapter ~/runs/run_a_r16/adapter
  sbatch slurm/serve_bench.sbatch --adapter ~/runs/ablation_answer_only/adapter

The eval loop's tokens/sec (evaluate.py, batch 4) is not a serving number: a served request arrives alone, and
batching hides per-request latency. Here each request goes through the whole path a served request takes:
build the prompt, preprocess the image, prefill, decode. Reported per request and summarised:

  e2e_s         preprocess + generate: what a caller waits for, minus HTTP
  ttft_s        time to first token after preprocessing (prefill)
  decode_tok_s  tokens/sec after the first token
  overall tok/s all new tokens / all generate seconds

The same N_REQUESTS val items (seed BENCH_SEED, after WARMUP excluded requests) for every adapter, so Run A and
the answer-only ablation are timed on identical inputs, and the latency trade is read beside the accuracy on
those same items. val, never test: this is a timing sample. The merged checkpoint (~7 GB) is deleted when the
job ends unless --keep-merged. Task 4.3 reruns the same requests through FastAPI; the difference is the HTTP cost.
"""
import argparse
import json
import os
import random
import shutil
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
import evaluate as ev  # noqa: E402  settings, data fingerprint, provenance, prompt building
import scoring  # noqa: E402
import serve_smoke  # noqa: E402  merge(), the same one the smoke test used

N_REQUESTS, WARMUP, BENCH_SEED = 50, 3, 71003
OUT = REPO / "results" / "serve"


def bench_items(items, n=N_REQUESTS, warmup=WARMUP, seed=BENCH_SEED):
    """(warm-up items, timed items): drawn by id, so every adapter gets the same requests in the same order."""
    ids = sorted(i["id"] for i in items)
    pick = random.Random(seed).sample(ids, n + warmup)
    by_id = {i["id"]: i for i in items}
    return [by_id[i] for i in pick[:warmup]], [by_id[i] for i in pick[warmup:]]


def pct(values, q):
    """Nearest-rank percentile: a value that was actually observed."""
    v = sorted(values)
    return v[max(0, min(len(v) - 1, -(-q * len(v) // 100) - 1))]


def summarise(rows):
    def dist(key):
        v = [r[key] for r in rows if r[key] is not None]
        return {"p50": round(pct(v, 50), 3), "p95": round(pct(v, 95), 3), "mean": round(sum(v) / len(v), 3)}
    tokens, gen = sum(r["new_tokens"] for r in rows), sum(r["generate_s"] for r in rows)
    return {"n": len(rows), "e2e_s": dist("e2e_s"), "ttft_s": dist("ttft_s"), "decode_tok_s": dist("decode_tok_s"),
            "overall_tok_s": round(tokens / gen, 1), "mean_new_tokens": round(tokens / len(rows), 1),
            "mean_prompt_tokens": round(sum(r["prompt_tokens"] for r in rows) / len(rows), 1),
            "accuracy_on_these_items": round(sum(r["correct"] for r in rows) / len(rows), 4),
            "schema_valid": round(sum(r["schema_valid"] for r in rows) / len(rows), 4)}


class Server:
    """The merged model behind one-request-at-a-time transformers.generate, timed the way a caller sees it."""

    def __init__(self, merged, s):
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor, StoppingCriteria
        self.torch, self.s = torch, s
        t0 = time.monotonic()
        self.processor = AutoProcessor.from_pretrained(merged, max_pixels=ev.st.MAX_PIXELS)
        self.net = AutoModelForImageTextToText.from_pretrained(merged, dtype=torch.bfloat16, device_map={"": 0})
        self.net.eval()
        self.load_seconds = round(time.monotonic() - t0, 1)

        class FirstToken(StoppingCriteria):   # called once per generated token; the first call times the prefill
            def __init__(self):
                self.t = None

            def __call__(self, input_ids, scores, **kw):
                if self.t is None:
                    torch.cuda.synchronize()
                    self.t = time.perf_counter()
                return torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)
        self.FirstToken = FirstToken

    def request(self, item, max_new_tokens):
        from transformers import StoppingCriteriaList
        torch, s = self.torch, self.s
        t0 = time.perf_counter()
        msgs, images = ev.build(item, [], REPO / s["data_dir"], REPO / s["items_dir"])
        batch = self.processor(text=[ev.prompt_text(self.processor, msgs)], images=images,
                               return_tensors="pt").to(self.net.device)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        first = self.FirstToken()
        with torch.inference_mode():
            gen = self.net.generate(**batch, max_new_tokens=max_new_tokens, do_sample=False,
                                    stopping_criteria=StoppingCriteriaList([first]))
        torch.cuda.synchronize()
        t2 = time.perf_counter()
        n_prompt = int(batch["input_ids"].shape[1])
        new = gen[0, n_prompt:]
        raw = self.processor.decode(new, skip_special_tokens=True)
        res = scoring.score(item, raw, s["style"])
        n_new = int(new.shape[0])
        return {"id": item["id"], "prompt_tokens": n_prompt, "new_tokens": n_new,
                "preprocess_s": round(t1 - t0, 4), "ttft_s": round(first.t - t1, 4) if first.t else None,
                "generate_s": round(t2 - t1, 4), "e2e_s": round(t2 - t0, 4),
                "decode_tok_s": round((n_new - 1) / (t2 - first.t), 1) if first.t and n_new > 1 and t2 > first.t else None,
                "correct": bool(res["correct"]), "schema_valid": bool(res["schema_valid"])}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--adapter", type=Path, required=True)
    ap.add_argument("--merged", type=Path, help="where to write the merged checkpoint (default ~/merged/<run>_bench)")
    ap.add_argument("--keep-merged", action="store_true")
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--dry-run", action="store_true", help="settings, data and provenance checks only; no GPU")
    ap.add_argument("--allow-unclean", action="store_true")
    args = ap.parse_args(argv)

    ns = ev.parse_args(["--adapter", str(args.adapter), "--split", "val"])
    s = ev.resolve(ns)
    items, _, fingerprint = ev.load_rows(s, "val", ns)
    prov, problems = ev.provenance(fingerprint, s["adapter"], s["data_dir"])
    warm, timed = bench_items(items)
    merged = (args.merged or Path.home() / "merged" / f"{s['run_name']}_bench").expanduser()
    out = args.out or OUT / f"{s['run_name']}.json"
    settings = {"engine": "transformers.generate", "precision": "bf16 merged", "batch_size": 1, "decoding": "greedy",
                "max_new_tokens": args.max_new_tokens, "targets": s["targets"], "style": s["style"],
                "split": "val", "seed": BENCH_SEED, "warmup": [i["id"] for i in warm], "items": [i["id"] for i in timed]}
    if args.dry_run:
        print(json.dumps({"run": s["run_name"], "merged": str(merged), "out": str(out), "settings": settings,
                          "provenance": prov}, indent=1))
        for p in problems:
            print(f"WARNING: the GPU job would refuse to start: {p}")
        print("SERVE_BENCH_DRY_RUN_OK")
        return 0
    if problems and not args.allow_unclean:
        sys.exit("refusing to start:\n  " + "\n  ".join(problems))

    import torch
    try:
        t0 = time.monotonic()
        serve_smoke.merge(Path(s["adapter"]), merged)
        merge_seconds = round(time.monotonic() - t0, 1)
        server = Server(merged, s)
        for item in warm:
            server.request(item, args.max_new_tokens)
        rows = []
        for i, item in enumerate(timed, 1):
            rows.append(server.request(item, args.max_new_tokens))
            print(f"  {i}/{len(timed)} {rows[-1]['e2e_s']:.2f}s {rows[-1]['new_tokens']} tok", flush=True)
    finally:
        if not args.keep_merged and merged.exists():
            shutil.rmtree(merged)
    report = {"task": "3.14", "run": s["run_name"], "adapter": s["adapter"], "settings": settings,
              "summary": summarise(rows), "load_seconds": server.load_seconds, "merge_seconds": merge_seconds,
              "gpu": torch.cuda.get_device_name(0), "peak_gpu_memory_mib": round(torch.cuda.max_memory_allocated() / 2**20),
              "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "versions": serve_smoke.versions(),
              "provenance": prov, "reportable": prov["clean"], "requests": rows}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(report["summary"], indent=1))
    print("SERVE_BENCH_OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
