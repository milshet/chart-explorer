#!/usr/bin/env python3
"""Task 4.1: merge the final adapter into bf16 base weights, then prove the merged checkpoint is the model that
was measured. Every Week 4 number (quantisation delta, served latency, red-team, model card) comes from these
weights, so an unverified merge means the card describes a model nobody evaluated.

  # login shell, no GPU: adapter, base and revision checks, the sample, provenance
  python scripts/serve/merge.py --adapter ~/runs/run_a_r16/adapter --dry-run
  # GPU, through Slurm: merge, verify, write the report
  sbatch slurm/merge.sbatch --adapter ~/runs/run_a_r16/adapter
  # re-verify an existing merged checkpoint without merging again
  sbatch slurm/merge.sbatch --adapter ~/runs/run_a_r16/adapter --verify-only

Three checks, in order. The first two raise; the third is a recorded verdict.

  1. base     adapter_config.json names the base the adapter was trained against, and run_summary.json the
              snapshot revision. Both must match the local base, or the merge raises. PEFT will happily merge onto
              other weights, and the result loads and generates fluently without being the model Week 2 measured.
              Settings the weight check cannot reproduce (DoRA, rank/alpha patterns, modules_to_save) raise too.
  2. weights  Read back from disk, the merged checkpoint must equal base + (alpha/r)·B@A, computed here from the
              adapter file itself rather than through PEFT, in exactly the modules the adapter names, and be
              bit-identical to the base everywhere else, in bf16 throughout. Merging folds the delta into each
              weight; a wrong dtype, scaling or module set drifts the output with nothing raising.
  3. answers  The merged checkpoint and the unmerged base+adapter answer the same SAMPLE_N val items through
              serve/generation.py, the path 4.2 and 4.3 also use. Compared on parsed answers, not raw strings.
              The tolerance (MAX_DISAGREE) is fixed here, before anything is measured.

Loaded in bf16, never 4-bit: folding full-precision deltas into quantised weights is neither the trained model nor
a clean quantisation of it. Writes the merged checkpoint (~7 GB, mind the 35 GB quota) with merge_manifest.json
beside the weights, which /health reports in 4.4, and results/serve/merge_<run>.json with every compared answer.
"""
import argparse
import datetime
import json
import os
import random
import re
import struct
import sys
import time
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1]
REPO = SCRIPTS.parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
import evaluate as ev  # noqa: E402  settings, data fingerprint, provenance
import train  # noqa: E402  sha256_file, git_state
from serve import generation as gen  # noqa: E402

# Fixed before any measurement. Changing them after seeing a result makes the verdict meaningless.
SAMPLE_N, SAMPLE_SEED = 100, 41001  # val items for the answer check: val, never test
MAX_DISAGREE = 3                    # parsed answers that may differ between merged and unmerged, of SAMPLE_N
MAX_SCHEMA_DROP = 1                 # schema-valid answers the merged model may lose against the reference
WEIGHT_MISMATCH_MAX = 0.01          # per module: share of the elements the merge should change that may match
                                    # neither correct rounding of W0 + delta (see check_merged_weights)
ULP_REL = 2 ** -7                   # bf16 has 8 significant bits: one unit in the last place is <= |x| * 2**-7

PEFT_PREFIX = "base_model.model."
MANIFEST = "merge_manifest.json"
OUT = REPO / "results" / "serve"


class MergeError(RuntimeError):
    """The merge would not produce, or did not produce, the model that was measured."""


# --- 1. which base, which adapter ----------------------------------------------------------------------------

def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8")) if Path(path).is_file() else None


def read_adapter(adapter):
    """Everything the checks need from an adapter directory and its run directory."""
    adapter = Path(adapter).expanduser().resolve()
    cfg = read_json(adapter / "adapter_config.json")
    if cfg is None:
        raise MergeError(f"{adapter} has no adapter_config.json")
    weights = adapter / "adapter_model.safetensors"
    if not weights.is_file():
        raise MergeError(f"{adapter} has no adapter_model.safetensors")
    summary = read_json(adapter.parent / "run_summary.json") or {}
    targets = read_json(adapter / "targets.json")
    return {"dir": adapter, "config": cfg, "weights": weights,
            "train_config": read_json(adapter / "train_config.json"),
            "trained_revision": summary.get("model_revision"),
            "targets": targets["modules"] if targets else None}


def base_id_and_revision(name_or_path):
    """(hub id, snapshot revision or None) from a hub id or an HF cache snapshot path."""
    m = re.search(r"models--([^/]+)--([^/]+)/snapshots/([0-9a-f]+)", str(name_or_path))
    if m:
        return f"{m.group(1)}/{m.group(2)}", m.group(3)
    return str(name_or_path).rstrip("/"), None


def lora_scaling(cfg):
    """The factor PEFT multiplies B@A by: alpha/r, or alpha/sqrt(r) with rsLoRA."""
    r, alpha = cfg["r"], cfg["lora_alpha"]
    return alpha / r ** 0.5 if cfg.get("use_rslora") else alpha / r


def check_base(a, local_revision, expected=ev.st.MODEL_ID, allow_unrecorded_revision=False):
    """Raise MergeError unless this adapter was trained against exactly the local base; return what was checked."""
    cfg = a["config"]
    if cfg.get("peft_type", "LORA") != "LORA":
        raise MergeError(f"peft_type {cfg.get('peft_type')}: only LoRA merges are verified")
    for key in ("use_dora", "modules_to_save", "rank_pattern", "alpha_pattern"):
        if cfg.get(key):
            raise MergeError(f"adapter_config.json sets {key}={cfg[key]!r}: the weight check computes plain "
                             "(alpha/r)·B@A and would not describe this merge")
    if cfg.get("fan_in_fan_out"):
        raise MergeError("fan_in_fan_out is set: not a Linear-layer LoRA")
    base_id, path_revision = base_id_and_revision(cfg.get("base_model_name_or_path", ""))
    if base_id != expected:
        raise MergeError(f"the adapter was trained against {cfg.get('base_model_name_or_path')!r}, "
                         f"but the merge base is {expected!r}")
    recorded = {"adapter_config.revision": cfg.get("revision"), "base_model_name_or_path": path_revision,
                "run_summary.model_revision": a.get("trained_revision")}
    for where, rev in recorded.items():
        if rev and rev != local_revision:
            raise MergeError(f"base revision mismatch: {where} records {rev}, the local base is {local_revision}")
    verified = any(recorded.values())
    if not verified and not allow_unrecorded_revision:
        raise MergeError("nothing records which base revision the adapter was trained against (no run_summary.json "
                         "beside the adapter): copy the run's run_summary.json next to adapter/, or pass "
                         "--allow-unrecorded-revision, which the manifest records")
    tc = a.get("train_config") or {}
    if tc.get("lora"):
        if (tc["lora"]["r"], tc["lora"]["alpha"]) != (cfg["r"], cfg["lora_alpha"]):
            raise MergeError(f"train_config.json says r={tc['lora']['r']} alpha={tc['lora']['alpha']}, "
                             f"adapter_config.json r={cfg['r']} alpha={cfg['lora_alpha']}: not this run's adapter")
    if tc.get("max_pixels") not in (None, ev.st.MAX_PIXELS):
        raise MergeError(f"the run used max_pixels={tc['max_pixels']}, serving loads images at {ev.st.MAX_PIXELS}")
    return {"base": expected, "base_revision": local_revision, "revision_verified": verified,
            "recorded_revisions": recorded, "r": cfg["r"], "alpha": cfg["lora_alpha"],
            "use_rslora": bool(cfg.get("use_rslora")), "scaling": lora_scaling(cfg)}


# --- 2. the adapter's factors, read from the file rather than through PEFT ------------------------------------

def safetensors_header(path):
    """{tensor name: {"dtype", "shape", ...}} from a .safetensors header: pure Python, no torch."""
    with open(path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(n))
    header.pop("__metadata__", None)
    return header


def factor_modules(keys):
    """{module name in the base model: {"A": key, "B": key}} from adapter file keys; raises on anything else."""
    mods, other = {}, []
    for k in keys:
        m = re.fullmatch(r"(.+)\.lora_([AB])(?:\.[^.]+)?\.weight", k)
        if not m:
            other.append(k)
            continue
        name = m.group(1)[len(PEFT_PREFIX):] if m.group(1).startswith(PEFT_PREFIX) else m.group(1)
        mods.setdefault(name, {})[m.group(2)] = k
    if other:
        raise MergeError(f"the adapter carries {len(other)} non-LoRA tensor(s), e.g. {other[:3]}: "
                         "the weight check covers LoRA factors only")
    half = sorted(n for n, ab in mods.items() if set(ab) != {"A", "B"})
    if half:
        raise MergeError(f"{len(half)} module(s) have only one LoRA factor, e.g. {half[:3]}")
    return mods


def check_targets(file_modules, targets):
    """The modules in the adapter file are exactly the ones training recorded wrapping (targets.json)."""
    if targets is None:
        return {"targets_json": False, "modules": len(file_modules)}
    recorded = {t[len(PEFT_PREFIX):] if t.startswith(PEFT_PREFIX) else t for t in targets}
    missing, extra = sorted(recorded - set(file_modules)), sorted(set(file_modules) - recorded)
    if missing or extra:
        raise MergeError(f"adapter file and targets.json disagree: {len(missing)} wrapped in training but absent "
                         f"from the file (e.g. {missing[:2]}), {len(extra)} in the file only (e.g. {extra[:2]})")
    parts = {}
    for n in file_modules:
        parts[train.module_part(n)] = parts.get(train.module_part(n), 0) + 1
    return {"targets_json": True, "modules": len(file_modules), "by_part": parts}


def check_merged_weights(before, after, factors, scaling):
    """Compare a merged state dict with base + scaling·B@A; raise MergeError listing every problem found.

    before, after: {name: tensor} for the base and the merged model, loaded the same way. factors: {module name:
    (A, B)}. A correct merge is W0 + delta rounded to the weights' dtype, either once (round(W0 + delta)) or the
    way PEFT does it on CPU (W0 + round(delta), the add itself rounded): an element matching either is correct.
    WEIGHT_MISMATCH_MAX allows for a different accumulation order in B@A; no element may be off by more than
    the rounding of W0 and delta can explain.
    """
    import torch
    problems, modules = [], {}
    targets = {f"{m}.weight": m for m in factors}
    absent = sorted(t for t in targets if t not in before)
    if absent:
        problems.append(f"{len(absent)} LoRA module(s) name no weight in the base, e.g. {absent[:2]}: "
                        "the base differs from the one the adapter was trained against")
    missing, extra = sorted(set(before) - set(after)), sorted(set(after) - set(before))
    if missing or extra:
        problems.append(f"merged keys differ from the base: {len(missing)} missing (e.g. {missing[:2]}), "
                        f"{len(extra)} extra (e.g. {extra[:2]})")
    untouched_changed, dtype_changed = [], []
    for name, w1 in after.items():
        w0 = before.get(name)
        if w0 is None:
            continue
        if w1.dtype != w0.dtype:
            dtype_changed.append(f"{name} {w0.dtype}->{w1.dtype}")
            continue
        if name not in targets:
            if not torch.equal(w0, w1):
                untouched_changed.append(name)
            continue
        a, b = factors[targets[name]]
        if tuple(w0.shape) != (b.shape[0], a.shape[1]):
            problems.append(f"{name}: base weight {tuple(w0.shape)} cannot take B@A {(b.shape[0], a.shape[1])}")
            continue
        delta = scaling * (b.float() @ a.float())
        exact = (w0.float() + delta).to(w0.dtype)           # rounded once
        stepwise = w0 + delta.to(w0.dtype)                  # PEFT: delta rounded, then added in the weights' dtype
        should_change = int((exact != w0).sum())
        mismatched = int(((w1 != exact) & (w1 != stepwise)).sum())
        bound = (w0.float().abs() + 2 * delta.abs()) * ULP_REL + torch.finfo(w0.dtype).tiny
        beyond_ulp = int(((w1.float() - exact.float()).abs() > bound).sum())
        moved = w1.float() - w0.float()
        dd = float((delta * delta).sum())
        modules[targets[name]] = {"elements": w0.numel(), "should_change": should_change, "mismatched": mismatched,
                                  "beyond_one_ulp": beyond_ulp,
                                  "coefficient": round(float((moved * delta).sum()) / dd, 4) if dd else None}
        if beyond_ulp or mismatched > WEIGHT_MISMATCH_MAX * max(should_change, 1):
            problems.append(f"{name}: {mismatched} of {should_change} elements differ from base + {scaling:g}·B@A "
                            f"({beyond_ulp} by more than rounding explains)")
    if dtype_changed:
        problems.append(f"{len(dtype_changed)} tensor(s) changed dtype in the merge, e.g. {dtype_changed[:2]}")
    if untouched_changed:
        problems.append(f"{len(untouched_changed)} tensor(s) outside the LoRA modules changed, "
                        f"e.g. {untouched_changed[:2]}")
    if problems:
        raise MergeError("merged weights are not base + LoRA delta:\n  " + "\n  ".join(problems))
    total = lambda k: sum(m[k] for m in modules.values())  # noqa: E731
    coefs = [m["coefficient"] for m in modules.values() if m["coefficient"] is not None]
    return {"modules_checked": len(modules), "tensors_unchanged": len(after) - len(modules),
            "dtype": str(next(iter(after.values())).dtype).replace("torch.", ""),
            "elements_in_modules": total("elements"), "elements_changed_by_merge": total("should_change"),
            "elements_rounded_differently": total("mismatched"),
            "coefficient_min": min(coefs) if coefs else None, "coefficient_max": max(coefs) if coefs else None,
            "per_module": modules}


# --- 3. the same answers ---------------------------------------------------------------------------------------

def sample_items(items, n=SAMPLE_N, seed=SAMPLE_SEED):
    """n items drawn by id, so a rerun compares the same requests in the same order."""
    by_id = {i["id"]: i for i in items}
    return [by_id[i] for i in random.Random(seed).sample(sorted(by_id), min(n, len(by_id)))]


def compare_answers(ref_rows, merged_rows, max_disagree=MAX_DISAGREE, max_schema_drop=MAX_SCHEMA_DROP):
    """Verdict on paired rows (same ids, same order): merged vs unmerged, on parsed answers."""
    if [r["id"] for r in ref_rows] != [r["id"] for r in merged_rows]:
        raise ValueError("reference and merged rows must cover the same items in the same order")
    pairs = list(zip(ref_rows, merged_rows))
    disagree = [r["id"] for r, m in pairs if not gen.same_answer(r["answer"], m["answer"])]
    count = lambda rows, k: sum(bool(r[k]) for r in rows)  # noqa: E731
    schema_ref, schema_merged = count(ref_rows, "schema_valid"), count(merged_rows, "schema_valid")
    passed = len(disagree) <= max_disagree and schema_merged >= schema_ref - max_schema_drop
    return {"n": len(pairs), "tolerance": {"max_disagree": max_disagree, "max_schema_drop": max_schema_drop},
            "disagree": len(disagree), "disagree_ids": disagree,
            "raw_identical": sum(r["raw"] == m["raw"] for r, m in pairs),
            "correct": {"reference": count(ref_rows, "correct"), "merged": count(merged_rows, "correct")},
            "schema_valid": {"reference": schema_ref, "merged": schema_merged},
            "fixed_by_merge": sum(m["correct"] and not r["correct"] for r, m in pairs),
            "broken_by_merge": sum(r["correct"] and not m["correct"] for r, m in pairs),
            "passed": passed}


def committed_agreement(ref_rows, predictions):
    """Informational: how often the batch-1 reference matches the committed batch-4 evaluation of the same model."""
    if not Path(predictions).is_file():
        return None
    committed = {}
    for line in open(predictions, encoding="utf-8"):
        p = json.loads(line)
        committed[p["id"]] = gen.parsed_answer(p["raw"])
    both = [r for r in ref_rows if r["id"] in committed]
    return {"file": str(Path(predictions).relative_to(REPO)), "n": len(both),
            "agree": sum(gen.same_answer(r["answer"], committed[r["id"]]) for r in both)}


# --- the job ----------------------------------------------------------------------------------------------------

def load_state(src):
    """(state dict on CPU, config) for a checkpoint, loaded in bf16 the way serving loads it."""
    import torch
    from transformers import AutoModelForImageTextToText
    model = AutoModelForImageTextToText.from_pretrained(str(src), dtype=torch.bfloat16, device_map={"": "cpu"},
                                                        local_files_only=True)
    if getattr(model.config, "quantization_config", None):
        raise MergeError(f"{src} loaded quantised: the merge is bf16 only")
    return model.state_dict(), model


def merge(adapter, merged_dir):
    """bf16 base + adapter -> merged checkpoint on disk (CPU: no GPU memory needed, nothing quantised)."""
    import torch
    from peft import PeftModel
    from transformers import AutoModelForImageTextToText, AutoProcessor
    model = AutoModelForImageTextToText.from_pretrained(ev.st.MODEL_ID, dtype=torch.bfloat16,
                                                        device_map={"": "cpu"}, local_files_only=True)
    model = PeftModel.from_pretrained(model, str(adapter), local_files_only=True).merge_and_unload()
    model.save_pretrained(merged_dir, safe_serialization=True)
    AutoProcessor.from_pretrained(str(adapter), local_files_only=True).save_pretrained(merged_dir)
    del model


def verify_weights(adapter, merged_dir, file_modules, scaling):
    from safetensors.torch import load_file
    tensors = load_file(str(adapter["weights"]))
    factors = {m: (tensors[ab["A"]], tensors[ab["B"]]) for m, ab in file_modules.items()}
    before, _ = load_state(ev.st.MODEL_ID)
    after, model = load_state(merged_dir)
    report = check_merged_weights(before, after, factors, scaling)
    del before, after, model, tensors, factors
    return report


def write_manifest(merged_dir, manifest):
    (Path(merged_dir) / MANIFEST).write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--adapter", type=Path, required=True, help="the final adapter, e.g. ~/runs/run_a_r16/adapter")
    ap.add_argument("--merged", type=Path, help="merged checkpoint directory (default ~/merged/<run_name>)")
    ap.add_argument("--verify-only", action="store_true", help="re-verify an existing --merged; do not merge")
    ap.add_argument("--out", type=Path, help=f"verification report (default {OUT.relative_to(REPO)}/merge_<run>.json)")
    ap.add_argument("--allow-unrecorded-revision", action="store_true",
                    help="no run_summary.json beside the adapter; recorded as revision_verified=false")
    ap.add_argument("--dry-run", action="store_true", help="checks 1 and the sample only; no GPU, nothing written")
    ap.add_argument("--allow-unclean", action="store_true")
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    t_start = time.monotonic()
    a = read_adapter(args.adapter)
    ns = ev.parse_args(["--adapter", str(a["dir"]), "--split", "val"])
    s = ev.resolve(ns)
    items, _, fingerprint = ev.load_rows(s, "val", ns)
    prov, problems = ev.provenance(fingerprint, s["adapter"], s["data_dir"])
    base = check_base(a, ev.st.model_revision(), allow_unrecorded_revision=args.allow_unrecorded_revision)
    file_modules = factor_modules(safetensors_header(a["weights"]))
    targets = check_targets(file_modules, a["targets"])
    sample = sample_items(items)
    merged_dir = (args.merged or Path.home() / "merged" / s["run_name"]).expanduser().resolve()
    out = args.out or OUT / f"merge_{s['run_name']}.json"
    plan = {"task": "4.1", "run": s["run_name"], "adapter": str(a["dir"]), "merged": str(merged_dir),
            "base_check": base, "targets": targets, "sample": {"split": "val", "n": len(sample), "seed": SAMPLE_SEED},
            "out": str(out)}
    if args.dry_run:
        print(json.dumps({**plan, "provenance": prov}, indent=1))
        for p in problems:
            print(f"WARNING: the GPU job would refuse to start: {p}")
        if not args.verify_only and merged_dir.exists():
            print(f"WARNING: {merged_dir} exists; the job refuses to overwrite it (use --verify-only, or remove it)")
        print("MERGE_DRY_RUN_OK")
        return 0
    if not os.getenv("SLURM_JOB_ID"):
        sys.exit("Submit with sbatch (slurm/merge.sbatch); --dry-run works on the login shell.")
    if problems and not args.allow_unclean:
        sys.exit("refusing to start:\n  " + "\n  ".join(problems))
    if args.verify_only and not merged_dir.is_dir():
        sys.exit(f"--verify-only: {merged_dir} does not exist")
    if not args.verify_only and merged_dir.exists():
        sys.exit(f"{merged_dir} exists: never overwritten. Re-verify it with --verify-only, or remove it first.")

    timings = {}
    if not args.verify_only:
        t0 = time.monotonic()
        print(f"merging {a['dir']} into bf16 {ev.st.MODEL_ID} -> {merged_dir}", flush=True)
        merge(a["dir"], merged_dir)
        timings["merge_s"] = round(time.monotonic() - t0, 1)
    manifest = {**{k: plan[k] for k in ("task", "run", "adapter", "merged")}, "status": "UNVERIFIED",
                "adapter_sha256": train.sha256_file(a["weights"]), "base_check": base, "targets": targets,
                "max_new_tokens": gen.MAX_NEW_TOKENS, "git_commit": prov["git_commit"],
                "created": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")}
    write_manifest(merged_dir, manifest)  # a job that dies from here on leaves the checkpoint marked unverified

    t0 = time.monotonic()
    print("checking merged weights against base + (alpha/r)·B@A", flush=True)
    weights = verify_weights(a, merged_dir, file_modules, base["scaling"])
    timings["weight_check_s"] = round(time.monotonic() - t0, 1)
    print(json.dumps({k: v for k, v in weights.items() if k != "per_module"}), flush=True)

    t0 = time.monotonic()
    print(f"answering {len(sample)} val items: unmerged base + adapter", flush=True)
    ref = gen.HFEngine(adapter=a["dir"])
    ref_rows = gen.answer_items(ref, sample, s)
    ref.close()
    print("answering the same items: merged checkpoint", flush=True)
    merged = gen.HFEngine(model_dir=merged_dir)
    merged_rows = gen.answer_items(merged, sample, s)
    engine = merged.description
    merged.close()
    timings["answer_check_s"] = round(time.monotonic() - t0, 1)
    answers = compare_answers(ref_rows, merged_rows)
    committed = committed_agreement(ref_rows, REPO / "results/evals" /
                                    f"{s['run_name']}-adapter_val_{s['variant']}_{s['style']}_0shot_bf16/predictions.jsonl")

    import torch
    status = "VERIFIED" if answers["passed"] else "FAILED_ANSWER_CHECK"
    files = {p.name: train.sha256_file(p) for p in sorted(merged_dir.glob("*.safetensors"))}
    summary_weights = {k: v for k, v in weights.items() if k != "per_module"}
    manifest.update(status=status, merged_files_sha256=files, weight_check=summary_weights,
                    answer_check={k: v for k, v in answers.items() if k != "disagree_ids"},
                    reportable=prov["clean"] and status == "VERIFIED")
    write_manifest(merged_dir, manifest)
    report = {**manifest, "engine": engine, "settings": {"split": "val", "seed": SAMPLE_SEED,
                                                         "items": [i["id"] for i in sample]},
              "weight_check": weights, "answer_check": answers, "reference_vs_committed_eval": committed,
              "timings": {**timings, "total_min": round((time.monotonic() - t_start) / 60, 1)},
              "gpu": torch.cuda.get_device_name(0), "peak_gpu_memory_mib": round(torch.cuda.max_memory_allocated() / 2**20),
              "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "provenance": prov,
              "rows": [{"id": r["id"], "answer_type": r["answer_type"], "reference": r, "merged": m}
                       for r, m in zip(ref_rows, merged_rows)]}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({"status": status, **{k: answers[k] for k in ("n", "disagree", "correct", "schema_valid")},
                      "reference_vs_committed_eval": committed}, indent=1))
    print(f"MERGE_{status} {merged_dir} -> {out}")
    return 0 if status == "VERIFIED" else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except MergeError as e:
        sys.exit(f"MERGE_REFUSED: {e}")
