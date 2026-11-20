#!/usr/bin/env python3
"""LLM-as-judge rubric (task 3.7): correctness, grounding and appropriate abstention, graded independently.

  python scripts/judge.py sample --name run_a_r16             # draw the judged sample; writes results/judge/<name>/sample.jsonl
  sbatch slurm/judge.sbatch --name run_a_r16                  # judge it locally on VM03 (default, free; task 3.13)
  python scripts/judge.py run --name run_a_r16 --backend anthropic   # or via the API (credentials; costs money)
  python scripts/judge.py spotcheck --name run_a_r16          # write judge_spot_check.jsonl for two human graders
  python scripts/judge.py agree --name run_a_r16              # judge vs humans: raw agreement AND Cohen's kappa

Three axes, graded separately because they come apart: an answer can be right with invented evidence, or well
grounded and wrong. Correctness and grounding are yes / no / n/a; abstention is graded on every item as yes / no
("was declining, or answering, the right call here").

The sample mixes Eval-A AND Eval-C: the abstention axis grades both directions, and Eval-A holds only nine
unanswerable items, so a sample from it alone would grade half the axis while the score reads as a whole one.

A verdict the parser cannot read, or a refusal from the judge, is EXCLUDED and counted (n_excluded), never scored
as zero: otherwise a judge whose output format drifted reads as a system that got worse. Grounding has a
programmatic cross-check this project already owns, the scorer's evidence_grounded; disagreements are listed.

Human agreement is reported as Cohen's kappa beside raw agreement: these axes are skewed, and 0.9 raw agreement
on a 90/10 split is what always-say-yes scores.

The judge reads text only (question, gold answer, the chart's data table, the model's output); it never sees the
image, so it grades against the table, which is the ground truth the labels came from.

The local judge (plan §9 decision 13) is arena-education/Llama-3.1-8B-Instruct-bnb-8bit: Llama-3.1-8B-Instruct stored in
bitsandbytes 8-bit, which loads on the locked transformers/bitsandbytes with nothing installed. It is not Qwen, so it
does not grade its own family. openai/gpt-oss-20b was planned and did not fit the L40S once expanded to bf16 (VM03
jobs 119, 120); its loading path (MXFP4 dequantised, final channel only) is kept for a machine where it does fit.
Greedy decoding. The summary records the judge model, its snapshot revision, the backend, the decoding and the
commit.
"""
import argparse
import collections
import json
import random
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
import rescore  # noqa: E402  items_for

OUT = REPO / "results" / "judge"
AXES = ("correctness", "grounding", "abstention")
VERDICTS = ("yes", "no", "n/a")
SAMPLE = {"test": 60, "eval_c": 20}   # cut rule: the judge sample size is the second thing cut if the week runs short
SEED = 61003
MODEL = "claude-opus-5-5"            # --backend anthropic
LOCAL_MODEL = "arena-education/Llama-3.1-8B-Instruct-bnb-8bit"   # --backend local (decision 13); gpt-oss-20b did not fit
LOCAL_MAX_NEW_TOKENS = 1024          # reasoning + verdict; a reply cut off before its verdict is excluded, and counted
REASONING_EFFORT = "low"
RUBRIC = """You are grading one answer from a chart question-answering system used in regulated financial reporting.
You see the question, the gold answer, the chart's underlying data table, and the system's raw output (JSON with
evidence, rationale, answer_type, unit, answer). Grade three axes independently:

- correctness: is the system's answer the gold answer (numbers within 5%)? "n/a" when the gold says the question
  is unanswerable.
- grounding: are the values the system cites as evidence actually in the data table (or correct computations from
  it)? "n/a" when it cites no evidence.
- abstention: was answering, or declining (answer null), the right call? The question is unanswerable exactly when
  the gold answer is null. Always "yes" or "no".

Reply with only a JSON object: {"correctness": "yes"|"no"|"n/a", "grounding": "yes"|"no"|"n/a",
"abstention": "yes"|"no", "note": "<one sentence>"}"""


def eval_folder(name, split):
    """The model's image-only evaluation on a split: the served condition, never a RAG one. `name` is the run
    (adapter evals are '<run>-adapter_...') and must match exactly, so run_a_r16 never picks up run_a_r16_table,
    run_a_r16_vision or a RAG folder. More than one match (the base model has several prompt settings) raises."""
    evals = REPO / "results/evals"
    matches = sorted(p.parent for pat in (f"{name}-adapter_{split}_image_only_*", f"{name}_{split}_image_only_*")
                     for p in evals.glob(f"{pat}/summary.json") if "_limit" not in p.parent.name)
    if len(matches) > 1:
        raise ValueError(f"{name} has {len(matches)} {split} evaluations; judge one: {[m.name for m in matches]}")
    return matches[0] if matches else None


def draw(name):
    """The judged sample: rows from Eval-A test and Eval-C for one model's committed predictions."""
    rows, missing = [], []
    for split, n in SAMPLE.items():
        folder = eval_folder(name, split)
        if folder is None:
            missing.append(split)
            continue
        settings = json.loads((folder / "summary.json").read_text(encoding="utf-8"))["settings"]
        items = rescore.items_for(settings)
        preds = [json.loads(l) for l in open(folder / "predictions.jsonl", encoding="utf-8")]
        rng = random.Random(f"{SEED}-{split}")
        una = [p for p in preds if p["answer_type"] == "unanswerable"]
        ans = [p for p in preds if p["answer_type"] != "unanswerable"]
        # every unanswerable test item (only nine exist), the rest answerable; Eval-C is all unanswerable
        picked = una + rng.sample(ans, min(len(ans), n - len(una))) if split == "test" else rng.sample(preds, min(n, len(preds)))
        for p in sorted(picked, key=lambda p: p["id"]):
            it = items[p["id"]]
            gold = json.loads(it["messages"][2]["content"])["answer"]
            rows.append({"id": p["id"], "split": split, "eval": folder.name, "question": it["question"], "gold": gold,
                         "table": it["input_text"], "raw": p["raw"], "scorer_correct": p["correct"],
                         "scorer_evidence": [p["evidence_ok"], p["evidence_n"]]})
    return rows, missing


def prompt(row):
    return (f"Question: {row['question']}\nGold answer: {json.dumps(row['gold'])}\n\nData table:\n{row['table']}\n\n"
            f"System output:\n{row['raw']}")


def parse(text):
    """A verdict dict, or None when the reply is not a readable verdict (excluded, never scored zero)."""
    sys.path.insert(0, str(REPO / "scripts"))
    import scoring
    v = scoring.extract_json(text or "")
    if not isinstance(v, dict):
        return None
    out = {a: str(v.get(a, "")).strip().lower() for a in AXES}
    if any(out[a] not in VERDICTS for a in AXES) or out["abstention"] == "n/a":
        return None
    return {**out, "note": v.get("note")}


def call_judge(client, row, model):
    response = client.beta.messages.create(
        model=model, max_tokens=16000, system=RUBRIC,
        output_config={"effort": "medium"},                # set explicitly: the default differs by model
        betas=["server-side-fallback-2026-07-01"], fallbacks="default",
        messages=[{"role": "user", "content": prompt(row)}])
    if response.stop_reason == "refusal":
        return None, "refusal"
    text = "".join(b.text for b in response.content if b.type == "text")
    return text, response._request_id


def final_text(text):
    """The final answer of a reply. gpt-oss writes an analysis channel, then a final one; only the final is the
    verdict (its reasoning may quote JSON). Replies with no channel markers are returned whole."""
    marker = "<|channel|>final<|message|>"
    if marker in text:
        text = text.rsplit(marker, 1)[1]
    elif "<|channel|>" in text:
        return ""                                       # reasoning only: cut off before it reached a verdict
    for end in ("<|return|>", "<|end|>", "<|endoftext|>"):
        text = text.split(end, 1)[0]
    return text.strip()


def mxfp4(model_id):
    """Whether a model's checkpoint ships MXFP4-quantised weights (gpt-oss does), read from its config.json."""
    from transformers import AutoConfig
    q = getattr(AutoConfig.from_pretrained(model_id), "quantization_config", None) or {}
    return (q.get("quant_method") if isinstance(q, dict) else getattr(q, "quant_method", None)) == "mxfp4"


class LocalJudge:
    """A Hugging Face chat model on the local GPU: loaded once, greedy, chat template applied by the tokenizer."""

    def __init__(self, model_id):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from rag.retrieve import snapshot
        self.torch, self.model_id, self.revision = torch, model_id, snapshot(model_id)
        self.tok = AutoTokenizer.from_pretrained(model_id)
        kw, self.load_notes = {}, {}
        if mxfp4(model_id):
            # Without the MXFP4 kernels transformers expands the experts to bf16 itself, and left to infer that
            # (VM03 job 119) it kept them on the CPU while the rest went to the GPU. Ask for it explicitly.
            from transformers import Mxfp4Config
            kw = {"quantization_config": Mxfp4Config(dequantize=True), "dtype": torch.bfloat16}
            self.load_notes["quantization"] = "mxfp4 dequantized to bf16"
        self.model = AutoModelForCausalLM.from_pretrained(model_id, device_map="cuda", **{"dtype": "auto", **kw})
        stray = sorted({str(t.device) for t in (*self.model.parameters(), *self.model.buffers())} - {"cuda:0"})
        if stray:   # anything still off the GPU is moved, and recorded: a silent CPU expert is a 100x slowdown
            self.load_notes["moved_to_cuda_from"] = stray
            self.model.to("cuda:0")
        self.model.eval()

    def __call__(self, row):
        msgs = [{"role": "system", "content": RUBRIC}, {"role": "user", "content": prompt(row)}]
        # reasoning_effort is read by gpt-oss's chat template and ignored by templates that do not use it
        ids = self.tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt", return_dict=True,
                                           reasoning_effort=REASONING_EFFORT).to(self.model.device)
        with self.torch.inference_mode():
            out = self.model.generate(**ids, do_sample=False, max_new_tokens=LOCAL_MAX_NEW_TOKENS)
        new = out[0, ids["input_ids"].shape[1]:]
        return self.tok.decode(new, skip_special_tokens=False), {"new_tokens": int(new.shape[0])}


def judge_rows(rows, ask):
    """Every sampled row with the judge's raw reply and its verdict; ask(row) -> (raw text or None, extra dict)."""
    judged = []
    for i, r in enumerate(rows, 1):
        raw, extra = ask(r)
        judged.append({**r, "judge_raw": raw, **extra, "verdict": parse(final_text(raw)) if raw else None})
        print(f"  {i}/{len(rows)} {r['id']} {'ok' if judged[-1]['verdict'] else 'EXCLUDED'}", flush=True)
    return judged


def commit():
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True).stdout.strip()
    except OSError:
        return None


def summarise(judged):
    ok = [j for j in judged if j["verdict"]]
    out = {"n": len(judged), "n_excluded": len(judged) - len(ok)}
    for a in AXES:
        graded = [j["verdict"][a] for j in ok if j["verdict"][a] != "n/a"]
        out[a] = {"yes": sum(v == "yes" for v in graded), "n": len(graded)}
    # the scorer's grounding vs the judge's, where both have an opinion
    out["grounding_disagreements"] = [j["id"] for j in ok if j["verdict"]["grounding"] != "n/a" and j["scorer_evidence"][1]
                                      and (j["verdict"]["grounding"] == "yes") != (j["scorer_evidence"][0] == j["scorer_evidence"][1])]
    exact = [j for j in ok if "gold" in j and "raw" in j]
    if exact:
        out["abstention_vs_exact"] = abstention_vs_exact(exact)
        # the rubric says correctness is n/a when the gold is null; grading it anyway is a judge error, counted
        out["correctness_graded_when_gold_null"] = sum(j["gold"] is None and j["verdict"]["correctness"] != "n/a"
                                                       for j in exact)
    return out


def declined(raw):
    import scoring
    v = scoring.extract_json(raw or "")
    return isinstance(v, dict) and v.get("answer") is None


def abstention_vs_exact(judged):
    """The judge's accuracy on the one axis whose answer is exact. Declining was the right call if and only if the
    gold answer is null, so the judge's abstention verdict can be checked row by row without a human. Broken down by
    case, because the failure that matters is which way it errs: e.g. marking a right call wrong because the answer
    or its evidence was wrong is leakage from the other two axes, which 3.7 requires be graded independently."""
    cases = collections.Counter()
    wrong = []
    for j in judged:
        should, did = j["gold"] is None, declined(j["raw"])
        right_call = should == did
        says_yes = j["verdict"]["abstention"] == "yes"
        case = f"{'unanswerable' if should else 'answerable'}, {'declined' if did else 'answered'}"
        cases[f"{case}: judge {'yes' if says_yes else 'no'}"] += 1
        if says_yes != right_call:
            wrong.append(j["id"])
    return {"agree": len(judged) - len(wrong), "n": len(judged), "by_case": dict(sorted(cases.items())),
            "disagreements": wrong}


def cohen(a, b):
    """(n, raw agreement, Cohen's kappa) over items both raters graded."""
    pairs = [(x, y) for x, y in zip(a, b) if x and y]
    n = len(pairs)
    if not n:
        return 0, None, None
    po = sum(x == y for x, y in pairs) / n
    ca, cb = collections.Counter(x for x, _ in pairs), collections.Counter(y for _, y in pairs)
    pe = sum(ca[k] * cb[k] for k in set(ca) | set(cb)) / n / n
    return n, round(po, 3), (round((po - pe) / (1 - pe), 3) if pe < 1 else None)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=("sample", "run", "summarise", "spotcheck", "agree"),
                    help="summarise: recompute summary.json's counts from a committed judged.jsonl, no model")
    ap.add_argument("--name", required=True, help="evaluation folder prefix, e.g. run_a_r16")
    ap.add_argument("--backend", choices=("local", "anthropic"), default="local")
    ap.add_argument("--model", help=f"default: {LOCAL_MODEL} (local) or {MODEL} (anthropic)")
    ap.add_argument("--dry-run", action="store_true", help="run step, local backend: render the prompts, no GPU")
    args = ap.parse_args()
    args.model = args.model or (LOCAL_MODEL if args.backend == "local" else MODEL)
    d = OUT / args.name
    d.mkdir(parents=True, exist_ok=True)
    if args.step == "sample":
        rows, missing = draw(args.name)
        (d / "sample.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8", newline="\n")
        print(f"{len(rows)} rows; not yet evaluated, so not sampled: {missing or 'none'}")
    elif args.step == "run":
        rows = [json.loads(l) for l in open(d / "sample.jsonl", encoding="utf-8")]
        if args.dry_run:   # login shell, no GPU: the chat template renders and the prompts fit
            from transformers import AutoTokenizer
            tok = AutoTokenizer.from_pretrained(args.model)
            texts = [tok.apply_chat_template([{"role": "system", "content": RUBRIC}, {"role": "user", "content": prompt(r)}],
                                             add_generation_prompt=True, tokenize=False, reasoning_effort=REASONING_EFFORT)
                     for r in rows]
            lengths = [len(tok(t, add_special_tokens=False)["input_ids"]) for t in texts]
            print(texts[0][-600:])
            print(f"\n{len(rows)} prompts, {min(lengths)}-{max(lengths)} tokens; reasoning effort "
                  f"{'in template' if 'Reasoning: ' + REASONING_EFFORT in texts[0] else 'NOT in template'}")
            print("JUDGE_DRY_RUN_OK")
            return
        t0 = time.monotonic()
        if args.backend == "local":
            ask = LocalJudge(args.model)
            meta = {"judge_revision": ask.revision, "decoding": "greedy", "max_new_tokens": LOCAL_MAX_NEW_TOKENS,
                    "reasoning_effort": REASONING_EFFORT, **ask.load_notes}
            print(f"loaded {args.model} {ask.load_notes or ''}", flush=True)
        else:
            import anthropic
            client = anthropic.Anthropic()

            def ask(r):
                text, req = call_judge(client, r, args.model)
                return text, {"request_id": req}
            meta = {"effort": "medium"}
        judged = judge_rows(rows, ask)
        (d / "judged.jsonl").write_text("".join(json.dumps(j, ensure_ascii=False) + "\n" for j in judged), encoding="utf-8", newline="\n")
        s = {"judge_model": args.model, "backend": args.backend, **meta, "commit": commit(),
             "minutes": round((time.monotonic() - t0) / 60, 1), **summarise(judged)}
        if args.backend == "local":
            s["peak_gpu_memory_mib"] = round(ask.torch.cuda.max_memory_allocated() / 2**20)
        (d / "summary.json").write_text(json.dumps(s, indent=1) + "\n", encoding="utf-8", newline="\n")
        print(json.dumps(s, indent=1))
    elif args.step == "summarise":   # the judge's run metadata is kept; only the counts are recomputed
        judged = [json.loads(l) for l in open(d / "judged.jsonl", encoding="utf-8")]
        s = json.loads((d / "summary.json").read_text(encoding="utf-8"))
        s.update(summarise(judged))
        (d / "summary.json").write_text(json.dumps(s, indent=1) + "\n", encoding="utf-8", newline="\n")
        print(json.dumps({k: s[k] for k in ("n", "n_excluded", "abstention_vs_exact", "correctness_graded_when_gold_null")
                          if k in s}, indent=1))
    elif args.step == "spotcheck":
        judged = [json.loads(l) for l in open(d / "judged.jsonl", encoding="utf-8")]
        pick = random.Random(SEED).sample(judged, min(30, len(judged)))
        blank = {"grader_1": {a: None for a in AXES}, "grader_2": {a: None for a in AXES}}
        (d / "judge_spot_check.jsonl").write_text("".join(json.dumps({k: j[k] for k in ("id", "question", "gold", "table", "raw")} | blank,
                                                                     ensure_ascii=False) + "\n" for j in pick), encoding="utf-8", newline="\n")
        print(f"wrote {len(pick)} items; two people fill grader_1 and grader_2 independently")
    else:
        judged = {j["id"]: j for j in map(json.loads, open(d / "judged.jsonl", encoding="utf-8"))}
        spot = [json.loads(l) for l in open(d / "judge_spot_check.jsonl", encoding="utf-8")]
        report = {}
        for a in AXES:
            j = [(judged[s["id"]]["verdict"] or {}).get(a) for s in spot]
            g1, g2 = [s["grader_1"][a] for s in spot], [s["grader_2"][a] for s in spot]
            report[a] = {"human_vs_human": cohen(g1, g2), "judge_vs_grader_1": cohen(j, g1), "judge_vs_grader_2": cohen(j, g2)}
        (d / "agreement.json").write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8", newline="\n")
        print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
