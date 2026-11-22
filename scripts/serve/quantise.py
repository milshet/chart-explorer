#!/usr/bin/env python3
"""Task 4.2: quantise the verified merge's language tower to 4 bits, keep the vision tower in bf16, and decide by a
measured accuracy delta whether to serve it.

  python scripts/serve/quantise.py --dry-run                   # login shell: every check that needs no GPU
  sbatch slurm/quantise.sbatch                                  # quantise, inspect, score both, verdict

Why the vision tower stays bf16: it turns pixels into the features every answer is read from, it is a small share of
the parameters, and chart questions turn on fine distinctions. Quantising it saves little and risks what the model
is for. Four guards, each a check rather than a convention:

  1. skip patterns   are applied to this checkpoint's own module names before anything runs: every pattern must
                     match something, every vision-tower and merger module must be kept, and the modules to
                     quantise must be exactly the 252 decoder projections. A pattern that matches nothing would
                     quantise the vision tower while the config says it does not, and the symptom would look like
                     ordinary quantisation loss.
  2. calibration     CALIB_N train conversations, checked disjoint from val, test, Eval-C and the RAG sets by id,
                     chart and image bytes. No gradient step is taken, which makes a leak easy to wave through, but
                     the quantisation grid is fitted to preserve activations on exactly these inputs.
  3. the export      is read back: every decoder projection must carry quantised tensors, and no vision-tower, merger
                     or lm_head weight may be quantised. Their stored dtype is recorded (GPTQ tooling may save
                     unquantised modules in fp16; the plan's own wording is "vision tower left at fp16").
  4. the verdict     is a delta on identical rows, the headline split (test, Eval-A), through serve/generation.py, the
                     path 4.1 verified and 4.3 serves. Both models are scored in this job. The tolerance is fixed
                     below, before any measurement: ACCEPT if accuracy drops by at most MAX_ACCURACY_DROP and the
                     quantised model loses at most MAX_VALID_DROP contract-valid answers; otherwise REJECT, and the
                     plan's fallback applies: serve bf16.

The export and the delta both run in this one pass: a step that only printed instructions would produce no model
and no delta. A backend that is absent or refuses the VL checkpoint is a recorded downgrade (status
DOWNGRADE_BACKEND_FAILED, the exception kept in the report), never a silent skip. Writes the quantised checkpoint
(with quant_manifest.json) and results/serve/quantise_<run>.json.
"""
import argparse
import datetime
import hashlib
import json
import math
import os
import random
import re
import sys
import time
import traceback
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1]
REPO = SCRIPTS.parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
import evaluate as ev  # noqa: E402
import scoring  # noqa: E402
import train  # noqa: E402
from serve import generation as gen  # noqa: E402
from serve import merge as mg  # noqa: E402  safetensors_header(), the merge manifest

# Fixed before any measurement.
BITS, GROUP_SIZE = 4, 128
CALIB_N, CALIB_SEED = 128, 42002
MAX_ACCURACY_DROP = 0.02    # absolute, on test (180 items: at most 3 more wrong answers than bf16)
MAX_VALID_DROP = 1          # contract-valid answers the quantised model may lose against bf16, of the 180
HEADLINE_SPLIT = "test"

KEEP = {"vision tower and merger": r"(^|\.)visual\.", "lm_head": r"(^|\.)lm_head$"}   # never quantised
DECODER_PROJ = re.compile(r"(^|\.)layers\.\d+\.(self_attn\.(q|k|v|o)_proj|mlp\.(gate|up|down)_proj)$")
EXPECTED_DECODER_PROJ = 36 * 7      # Qwen2.5-VL-3B: 36 decoder layers x 7 projections
QUANT_SUFFIXES = ("qweight", "qzeros", "scales", "g_idx", "weight_packed", "weight_scale", "weight_zero_point",
                  "weight_shape", "weight_g_idx")
EVAL_SETS = [("data/v2.0", "val"), ("data/v2.0", "test"), ("data/eval_c", "eval_c"),
             *[("data/rag", n) for n in ("rag_val", "rag_test", "rag_test_qo", "rag_test_withheld")]]
MANIFEST = "quant_manifest.json"
OUT = REPO / "results" / "serve"


class QuantError(RuntimeError):
    """The quantisation would not be, or is not, the one the plan describes."""


# --- 1. which modules ---------------------------------------------------------------------------------------------

def checkpoint_tensors(model_dir):
    """{tensor name: header entry} over every .safetensors file in a checkpoint directory."""
    files = sorted(Path(model_dir).glob("*.safetensors"))
    if not files:
        raise QuantError(f"{model_dir} has no .safetensors files")
    out = {}
    for f in files:
        out.update(mg.safetensors_header(f))
    return out


def linear_modules(tensors):
    """Module names that own a 2-D .weight: the candidates any 4-bit scheme would quantise."""
    return sorted(n[:-len(".weight")] for n, h in tensors.items()
                  if n.endswith(".weight") and len(h.get("shape", [])) == 2)


def tied_embeddings(model_dir):
    """Whether the checkpoint's config ties lm_head to embed_tokens (then no lm_head tensor is saved at all)."""
    cfg = json.loads((Path(model_dir) / "config.json").read_text(encoding="utf-8"))
    for c in (cfg, cfg.get("text_config") or {}):
        if "tie_word_embeddings" in c:
            return bool(c["tie_word_embeddings"])
    return False


def plan_targets(modules, keep=KEEP, expected=EXPECTED_DECODER_PROJ, tied_lm_head=False):
    """{"quantise": [...], "keep": {...}} checked against this checkpoint's own names; raises QuantError.

    With tied embeddings (Qwen2.5-VL-3B: tie_word_embeddings=true) lm_head is embed_tokens and has no tensor of its
    own, so its pattern may match nothing; embed_tokens is then reported among the modules kept unquantised.
    """
    kept, problems = {}, []
    for label, pattern in keep.items():
        hits = [m for m in modules if re.search(pattern, m)]
        if not hits and not (label == "lm_head" and tied_lm_head):
            problems.append(f"skip pattern for {label} ({pattern}) matches no module of this checkpoint")
        kept[label] = hits
    kept_all = {m for hits in kept.values() for m in hits}
    quantise = [m for m in modules if DECODER_PROJ.search(m) and m not in kept_all]
    vision = [m for m in modules if ".visual." in f".{m}"]
    leaked = sorted(set(vision) & set(quantise))
    if leaked:
        problems.append(f"{len(leaked)} vision module(s) would be quantised, e.g. {leaked[:2]}")
    if vision and not set(vision) <= kept_all:
        problems.append("not every vision-tower module is covered by a skip pattern")
    if len(quantise) != expected:
        problems.append(f"{len(quantise)} decoder projections to quantise, expected {expected}")
    if problems:
        raise QuantError("quantisation targets are wrong for this checkpoint:\n  " + "\n  ".join(problems))
    other = [m for m in modules if m not in kept_all and m not in quantise]
    keep_counts = {k: len(v) for k, v in kept.items()}
    if tied_lm_head and not kept.get("lm_head"):
        keep_counts["lm_head"] = "tied to embed_tokens (no separate tensor)"
    return {"quantise": quantise, "keep": keep_counts, "other_linear_kept": other, "vision_modules": len(vision),
            "tied_lm_head": tied_lm_head}


def inspect_export(tensors, plan):
    """Read back the export: decoder projections quantised, vision tower and merger untouched in bf16."""
    by_module = {}
    for name, h in tensors.items():
        mod, _, suffix = name.rpartition(".")
        by_module.setdefault(mod, {})[suffix] = h
    quantised = {m for m, parts in by_module.items() if any(s in parts for s in QUANT_SUFFIXES)}
    problems = []
    missing = [m for m in plan["quantise"] if m not in quantised]
    if missing:
        problems.append(f"{len(missing)} decoder projection(s) were not quantised, e.g. {missing[:2]}")
    vision_q = sorted(m for m in quantised if ".visual." in f".{m}" or re.search(KEEP["lm_head"], m))
    if vision_q:
        problems.append(f"{len(vision_q)} module(s) that must stay bf16 were quantised, e.g. {vision_q[:2]}")
    vision_dtypes = {h["dtype"] for n, h in tensors.items() if ".visual." in f".{n}" and n.endswith(".weight")}
    if not vision_dtypes:
        problems.append("the export has no vision-tower weights")
    elif vision_dtypes - {"BF16", "F16", "F32"}:
        problems.append(f"vision-tower weights stored as {sorted(vision_dtypes)}: not unquantised floating point")
    if problems:
        raise QuantError("the export is not the planned quantisation:\n  " + "\n  ".join(problems))
    size = sum(math.prod(h["shape"]) * {"BF16": 2, "F16": 2, "F32": 4, "I32": 4, "I8": 1, "U8": 1, "I64": 8}.get(
        h["dtype"], 0) for h in tensors.values())
    return {"quantised_modules": len(quantised), "vision_weight_dtypes": sorted(vision_dtypes),
            "tensor_bytes": size}


# --- 2. calibration ---------------------------------------------------------------------------------------------

def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def image_digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def eval_fingerprints(sets=EVAL_SETS, variant="image_only"):
    """Ids, chart ids and image hashes of every evaluation set, read from the files on disk."""
    ids, charts, images = set(), set(), set()
    for data_dir, split in sets:
        for v in (variant, "image_table"):
            f = REPO / data_dir / v / f"{split}.jsonl"
            if f.is_file():
                for r in read_jsonl(f):
                    ids.add(r["id"])
                    charts.add(r["chart_id"])
                    images.add(image_digest(REPO / data_dir / r["image"]))
    if not ids:
        raise QuantError("found no evaluation set to check calibration against")
    return {"ids": ids, "charts": charts, "images": images}


def calibration_rows(train_rows, evals, n=CALIB_N, seed=CALIB_SEED, data_dir=REPO / "data/v2.0"):
    """n train conversations drawn by id, each checked disjoint from every evaluation set; raises on any overlap."""
    by_id = {r["id"]: r for r in train_rows}
    rows = [by_id[i] for i in random.Random(seed).sample(sorted(by_id), min(n, len(by_id)))]
    leaks = []
    for r in rows:
        if r["id"] in evals["ids"]:
            leaks.append(f"{r['id']}: id in an evaluation set")
        if r["chart_id"] in evals["charts"]:
            leaks.append(f"{r['id']}: chart {r['chart_id']} in an evaluation set")
        if image_digest(data_dir / r["image"]) in evals["images"]:
            leaks.append(f"{r['id']}: image bytes identical to an evaluation image")
    if leaks:
        raise QuantError(f"{len(leaks)} calibration row(s) overlap the evaluation sets, e.g. {leaks[:3]}")
    return rows


# --- 4. the verdict -------------------------------------------------------------------------------------------------

def mcnemar_exact(b, c):
    """Two-sided exact McNemar p-value for b vs c discordant pairs (the test every comparison in the report uses)."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def verdict(ref_rows, quant_rows, max_drop=MAX_ACCURACY_DROP, max_valid_drop=MAX_VALID_DROP):
    """ACCEPT or REJECT on paired rows (same items, same order), with every number the decision rests on."""
    if [r["id"] for r in ref_rows] != [r["id"] for r in quant_rows]:
        raise ValueError("bf16 and quantised rows must cover the same items in the same order")
    n = len(ref_rows)
    acc = lambda rows: sum(r["correct"] for r in rows) / n  # noqa: E731
    valid = lambda rows: sum(not scoring.contract_errors(scoring.extract_json(r["raw"]), r["style"])  # noqa: E731
                             for r in rows)
    pairs = list(zip(ref_rows, quant_rows))
    lost = sum(r["correct"] and not q["correct"] for r, q in pairs)
    gained = sum(q["correct"] and not r["correct"] for r, q in pairs)
    drop = acc(ref_rows) - acc(quant_rows)
    v_ref, v_q = valid(ref_rows), valid(quant_rows)
    accept = drop <= max_drop + 1e-12 and v_q >= v_ref - max_valid_drop
    return {"verdict": "ACCEPT" if accept else "REJECT", "n": n,
            "tolerance": {"max_accuracy_drop": max_drop, "max_valid_drop": max_valid_drop},
            "accuracy": {"bf16": round(acc(ref_rows), 4), "quantised": round(acc(quant_rows), 4),
                         "drop": round(drop, 4)},
            "contract_valid": {"bf16": v_ref, "quantised": v_q},
            "discordant": {"lost": lost, "gained": gained, "mcnemar_p": round(mcnemar_exact(lost, gained), 4)},
            "same_answer": sum(gen.same_answer(r["answer"], q["answer"]) for r, q in pairs),
            "mean_new_tokens": {"bf16": round(sum(r["new_tokens"] for r in ref_rows) / n, 1),
                                "quantised": round(sum(r["new_tokens"] for r in quant_rows) / n, 1)},
            "mean_seconds": {"bf16": round(sum(r["seconds"] for r in ref_rows) / n, 3),
                             "quantised": round(sum(r["seconds"] for r in quant_rows) / n, 3)}}


def downgrade(stage, exc):
    """A backend failure, recorded with its exception: the plan's fallback (serve bf16) applies."""
    return {"verdict": "REJECT", "status": "DOWNGRADE_BACKEND_FAILED", "stage": stage,
            "exception": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()[-4000:],
            "fallback": "serve the bf16 merge (task 4.1); 4.2 is dropped with this record as its reason"}


# --- backends ---------------------------------------------------------------------------------------------------------

def backend_versions(names=("gptqmodel", "llmcompressor", "compressed-tensors", "optimum", "autoawq")):
    from importlib.metadata import PackageNotFoundError, version
    out = {}
    for n in names:
        try:
            out[n] = version(n)
        except PackageNotFoundError:
            out[n] = None
    return out


def quantize_config_kwargs():
    """GPTQ settings: 4-bit, groups of 128, symmetric, no act-order (the kernels without nvcc support it best)."""
    return {"bits": BITS, "group_size": GROUP_SIZE, "sym": True, "desc_act": False}


def calibration_conversation(row, data_dir=REPO / "data/v2.0"):
    """One training conversation (system, user with the chart, assistant answer) in the chat format GPTQModel's
    Qwen2-VL definition takes. The image is the training-resized PIL image, so calibration sees training pixels."""
    sys_turn, user, assistant = row["messages"][:3]
    content = [{"type": "image", "image": ev.st.load_image(data_dir, row["image"]), "max_pixels": ev.st.MAX_PIXELS}
               if c["type"] == "image" else dict(c) for c in user["content"]]
    return [dict(sys_turn), {"role": "user", "content": content},
            {"role": "assistant", "content": assistant["content"]}]


def gptqmodel_backend(merged, out_dir, calib_rows, plan):
    """GPTQ with GPTQModel (chosen 2026-11-23: it installs against the lock; llm-compressor conflicts with it).

    GPTQModel's Qwen2-VL definition quantises model.language_model.layers.# {q,k,v,o,gate,up,down}_proj: the
    decoder projections plan_targets() expects. The export is checked independently by inspect_export().
    """
    import torch
    from gptqmodel import GPTQModel, QuantizeConfig
    from transformers import AutoProcessor
    convs = [calibration_conversation(r) for r in calib_rows]
    model = GPTQModel.load(str(merged), QuantizeConfig(**quantize_config_kwargs()))
    log = model.quantize(convs, batch_size=1)
    model.save(str(out_dir))
    AutoProcessor.from_pretrained(str(merged), local_files_only=True).save_pretrained(str(out_dir))
    del model
    torch.cuda.empty_cache()
    return {"backend": "gptqmodel", "quantize_config": quantize_config_kwargs(),
            "calibration_conversations": len(convs),
            "quantize_log_modules": len(log) if hasattr(log, "__len__") else None}


def backend_check(name):
    """Dry-run check that the backend imports and accepts the settings, so a typo cannot cost a GPU job."""
    if name == "gptqmodel":
        from gptqmodel import QuantizeConfig
        return repr(QuantizeConfig(**quantize_config_kwargs()))[:400]
    raise QuantError(f"unknown backend {name}")


BACKENDS = {"gptqmodel": gptqmodel_backend}  # name -> quantise(merged_dir, out_dir, calib_rows, plan) -> dict


# --- the job ------------------------------------------------------------------------------------------------------------

def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--merged", type=Path, default=Path.home() / "merged" / "run_a_r16", help="the verified bf16 merge")
    ap.add_argument("--out-dir", type=Path, help="quantised checkpoint (default <merged>_w4)")
    ap.add_argument("--backend", choices=sorted(BACKENDS), default="gptqmodel")
    ap.add_argument("--report", type=Path, help="default results/serve/quantise_<run>.json")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--allow-unclean", action="store_true")
    return ap.parse_args(argv)


def preflight(args):
    merged = args.merged.expanduser().resolve()
    manifest = json.loads((merged / mg.MANIFEST).read_text(encoding="utf-8"))
    if manifest.get("status") != "VERIFIED":
        raise QuantError(f"{merged}: merge status {manifest.get('status')!r}, not VERIFIED (task 4.1)")
    plan = plan_targets(linear_modules(checkpoint_tensors(merged)), tied_lm_head=tied_embeddings(merged))
    ns = ev.parse_args(["--adapter", manifest["adapter"], "--split", HEADLINE_SPLIT, "--final-test"])
    s = ev.resolve(ns)
    items, _, fingerprint = ev.load_rows(s, HEADLINE_SPLIT, ns)
    prov, problems = ev.provenance(fingerprint, s["adapter"], s["data_dir"])
    train_rows = train.read_split({k: s[k] for k in ("data_dir", "variant", "targets")}, "train")
    calib = calibration_rows(train_rows, eval_fingerprints())
    return merged, manifest, plan, s, items, prov, problems, calib


def main(argv=None):
    args = parse_args(argv)
    t_start = time.monotonic()
    merged, manifest, plan, s, items, prov, problems, calib = preflight(args)
    out_dir = (args.out_dir or merged.with_name(merged.name + f"_w{BITS}")).expanduser().resolve()
    report_path = args.report or OUT / f"quantise_{manifest['run']}.json"
    head = {"task": "4.2", "run": manifest["run"], "merged": str(merged), "out_dir": str(out_dir),
            "backend": args.backend, "scheme": {"bits": BITS, "group_size": GROUP_SIZE, "weights_only": True,
                                                "quantised": "decoder projections", "kept_bf16": list(KEEP)},
            "targets": {k: v for k, v in plan.items() if k != "quantise"} | {"quantise": len(plan["quantise"])},
            "calibration": {"split": "train", "n": len(calib), "seed": CALIB_SEED,
                            "disjoint_from": [f"{d}/{n}" for d, n in EVAL_SETS], "ids": [r["id"] for r in calib]},
            "headline": {"split": HEADLINE_SPLIT, "n": len(items)}, "backend_versions": backend_versions()}
    if args.dry_run:
        print(json.dumps({**{k: v for k, v in head.items() if k != "calibration"},
                          "calibration": {k: v for k, v in head["calibration"].items() if k != "ids"},
                          "provenance": prov}, indent=1))
        for p in problems:
            print(f"WARNING: the GPU job would refuse to start: {p}")
        try:
            print(f"backend {args.backend}: {backend_check(args.backend)}")
        except Exception as e:  # noqa: BLE001
            print(f"WARNING: backend {args.backend} is not usable here: {type(e).__name__}: {e}")
        conv = calibration_conversation(calib[0])
        print(f"calibration conversation: roles {[m['role'] for m in conv]}, image {conv[1]['content'][0]['image'].size}")
        if out_dir.exists():
            print(f"WARNING: {out_dir} exists; the job refuses to overwrite it")
        print("QUANTISE_DRY_RUN_OK")
        return 0
    if not os.getenv("SLURM_JOB_ID"):
        sys.exit("Submit with sbatch (slurm/quantise.sbatch); --dry-run works on the login shell.")
    if problems and not args.allow_unclean:
        sys.exit("refusing to start:\n  " + "\n  ".join(problems))
    if out_dir.exists():
        sys.exit(f"{out_dir} exists: never overwritten")

    import torch
    report, timings = dict(head), {}
    t0 = time.monotonic()
    try:
        if args.backend not in BACKENDS:
            raise QuantError("no quantisation backend is installed and wired in")
        report["backend_info"] = BACKENDS[args.backend](merged, out_dir, calib, plan)
        timings["quantise_s"] = round(time.monotonic() - t0, 1)
        report["export"] = inspect_export(checkpoint_tensors(out_dir), plan)
    except QuantError:
        raise  # a wrong export or wrong targets is not a backend failure: it must not pass as a downgrade
    except Exception as e:  # noqa: BLE001  the backend is absent, refuses the VL checkpoint, or fails mid-run
        report.update(downgrade("quantise", e))
    if "status" not in report:
        rows = {}
        for label, kw in (("bf16", {"model_dir": merged}), ("quantised", {"model_dir": out_dir, "allow_quantized": True})):
            t0 = time.monotonic()
            try:
                engine = gen.HFEngine(**kw)
            except Exception as e:  # noqa: BLE001  the quantised export does not load in the serving stack
                report.update(downgrade(f"load {label}", e))
                break
            print(f"answering {len(items)} {HEADLINE_SPLIT} items: {label}", flush=True)
            rows[label] = [{**r, "style": s["style"]} for r in gen.answer_items(engine, items, s)]
            engine.close()
            timings[f"answer_{label}_s"] = round(time.monotonic() - t0, 1)
        if "status" not in report:
            report.update(verdict(rows["bf16"], rows["quantised"]))
            report["status"] = "MEASURED"
            report["rows"] = [{"id": r["id"], "bf16": r, "quantised": q} for r, q in zip(rows["bf16"], rows["quantised"])]
    report.update(timings=timings | {"total_min": round((time.monotonic() - t_start) / 60, 1)},
                  gpu=torch.cuda.get_device_name(0), peak_gpu_memory_mib=round(torch.cuda.max_memory_allocated() / 2**20),
                  slurm_job_id=os.environ.get("SLURM_JOB_ID"), provenance=prov,
                  created=datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"))
    if out_dir.is_dir():
        (out_dir / MANIFEST).write_text(json.dumps({k: v for k, v in report.items() if k != "rows"}, indent=1) + "\n",
                                        encoding="utf-8")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({k: report.get(k) for k in ("status", "verdict", "accuracy", "contract_valid", "discordant",
                                                 "exception")}, indent=1))
    print(f"QUANTISE_{report['status']}_{report['verdict']} -> {report_path}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except QuantError as e:
        sys.exit(f"QUANTISE_REFUSED: {e}")
