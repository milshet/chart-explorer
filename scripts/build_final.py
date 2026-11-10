#!/usr/bin/env python3
"""Build the final ChartAudit-Fin v2.0 splits from the rule-based selection + blind labels.

Inputs
  --selection  data/v2.0/selection  (train/val/test.jsonl from select_clean.py, images/)
  --runs       folder with v2_train/ and v2_val/ blind-labelling runs (key/, outputs/, compare.jsonl)
Outputs (in --out)
  image_only/{train,val,test}.jsonl, image_table/..., each with answer_only/ copies,
  dropped_{train,val}.jsonl, stats.json, MANIFEST.txt

Train/val keep items whose final confidence is high or medium. The assistant target is
{"evidence", "rationale", "answer_type", "unit", "answer"} in that order: evidence and rationale
come from the blind label, the answer is the human ChartQA answer. Test is never labelled:
its target holds only the human answer.
"""
import argparse, collections, hashlib, json, os, re

SYSTEM = ("You are a chart analyst. Answer questions about the chart image provided.\n"
          "Return ONLY valid JSON matching the schema. Write the evidence and rationale first, then the answer.\n"
          'If the chart cannot answer the question, set "answer" to null and explain in "rationale".')
SCHEMA = ('{"evidence": [string], "rationale": string, '
          '"answer_type": "number" | "text" | "yes_no" | "unanswerable", "unit": string | null, '
          '"answer": string | number | null}')
SYSTEM_SHORT = ("You are a chart analyst. Answer questions about the chart image provided.\n"
                "Return ONLY valid JSON matching the schema. If the chart cannot answer the question,\n"
                'set "answer" to null.')
SCHEMA_SHORT = ('{"answer_type": "number" | "text" | "yes_no" | "unanswerable", "unit": string | null, '
                '"answer": string | number | null}')
TABLE = ("chart image provided", "chart image and its data table provided")


def nums(s):
    return [float(x.replace(",", "")) for x in re.findall(r"(?<![A-Za-z\d.])\d[\d,]*(?:\.\d+)?", str(s))]


def grounded(evidence, table):
    """Share of numbers cited in the evidence that appear in the chart's data table (within
    rounding). The labeller never saw this table. Computed values (sums, differences) and
    estimates read off an axis legitimately lower the share, so this is a flag, not a filter.
    None when the evidence cites no numbers."""
    tv = nums(table)
    vs = [v for e in evidence or [] for v in nums(e)]
    if not vs:
        return None
    hit = sum(any(abs(v - t) <= max(0.051, abs(t) * 0.01) for t in tv) for v in vs)
    return round(hit / len(vs), 3)


def answer_out(r):
    """Numbers as normalised numbers, text as the human label (original case), yes/no capitalised."""
    a = r["answer"]
    if r["answer_type"] == "text":
        return str(r["gold_answer_raw"]).strip().rstrip(".")
    if r["answer_type"] == "yes_no" and isinstance(a, str):
        return a.capitalize()
    return a


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def record(r, target, variant, conf, ground, short=False):
    q = f"Question: {r['question']}\n\nRespond with JSON: {SCHEMA_SHORT if short else SCHEMA}"
    system = SYSTEM_SHORT if short else SYSTEM
    if variant == "image_table":
        system = system.replace(*TABLE)
    if variant == "image_table":
        q = f"Chart data:\n{r['chart_data_table']}\n\n" + q
    content = [{"type": "image", "image": r["image"]}, {"type": "text", "text": q}]
    out = {k: r[k] for k in ("id", "split", "chart_id", "question_source_chart", "source", "chart_type", "question",
                             "gold_answer_raw", "answer", "answer_type", "question_type", "domain", "domain_tags",
                             "flags", "image")}
    out["input_text"] = r["chart_data_table"]   # chart data (table, colours); used only for scoring evidence in image_only
    out.update({"version": "2.0", "variant": variant, "confidence": conf, "evidence_in_table_share": ground,
                "messages": [{"role": "system", "content": system},
                             {"role": "user", "content": content},
                             {"role": "assistant", "content": json.dumps(target, ensure_ascii=False)}]})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selection", required=True)
    ap.add_argument("--runs", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    stats, manifest = {"version": "2.0", "splits": {}}, {}
    for split in ("train", "val", "test"):
        sel = [json.loads(l) for l in open(os.path.join(a.selection, f"{split}.jsonl"), encoding="utf-8")]
        keep, dropped, st = [], [], collections.Counter()
        if split == "test":
            for r in sel:
                tgt = {"evidence": [], "rationale": None, "answer_type": r["answer_type"], "unit": None,
                       "answer": answer_out(r)}
                keep.append((r, tgt, None, None))
        else:
            run = os.path.join(a.runs, f"v2_{split}")
            key = {k["pack_id"]: k["id"] for k in map(json.loads, open(os.path.join(run, "key", "answers.jsonl"), encoding="utf-8"))}
            labs = {}
            for f in sorted(os.listdir(os.path.join(run, "outputs"))):
                for l in open(os.path.join(run, "outputs", f), encoding="utf-8"):
                    if l.strip():
                        x = json.loads(l); labs[key[x["pack_id"]]] = x
            comp = {c["id"]: c for c in map(json.loads, open(os.path.join(run, "compare.jsonl"), encoding="utf-8"))}
            for r in sel:
                c, lab = comp.get(r["id"]), labs.get(r["id"])
                if c is None or lab is None:
                    st["no_label"] += 1; dropped.append({"id": r["id"], "reason": "no_label"}); continue
                g = grounded(lab.get("evidence"), r["chart_data_table"])
                st[f"conf_{c['final_conf']}"] += 1
                if c["final_conf"] == "low":
                    dropped.append({"id": r["id"], "reason": "blind_answer_disagrees", "gold": r["answer"],
                                    "blind": lab.get("answer"), "rationale": lab.get("rationale")})
                    continue
                if g is not None:
                    st["evidence_items_with_numbers"] += 1
                    st["evidence_all_numbers_in_table"] += g == 1.0
                    st["evidence_most_numbers_in_table"] += g >= 0.5
                un = r["answer_type"] == "unanswerable"
                tgt = {"evidence": [] if un else lab.get("evidence") or [], "rationale": lab.get("rationale"),
                       "answer_type": r["answer_type"], "unit": None if un else lab.get("unit"),
                       "answer": answer_out(r)}
                keep.append((r, tgt, c["final_conf"], g))
        for variant in ("image_only", "image_table"):
            for sub in ("", "answer_only"):
                d = os.path.join(a.out, variant, sub); os.makedirs(d, exist_ok=True)
                p = os.path.join(d, f"{split}.jsonl")
                with open(p, "w", encoding="utf-8", newline="\n") as f:
                    for r, tgt, conf, g in keep:
                        t = {"answer_type": tgt["answer_type"], "unit": tgt["unit"], "answer": tgt["answer"]} if sub else tgt
                        f.write(json.dumps(record(r, t, variant, conf, g, short=bool(sub)), ensure_ascii=False) + "\n")
                manifest[os.path.relpath(p, a.out)] = sha(p)
        if dropped:
            with open(os.path.join(a.out, f"dropped_{split}.jsonl"), "w", encoding="utf-8", newline="\n") as f:
                for x in dropped:
                    f.write(json.dumps(x, ensure_ascii=False) + "\n")
        stats["splits"][split] = {"selected": len(sel), "kept": len(keep), "dropped": len(dropped), **st,
                                  "by_answer_type": dict(collections.Counter(r["answer_type"] for r, *_ in keep)),
                                  "by_question_type": dict(collections.Counter(r["question_type"] for r, *_ in keep))}
    json.dump(stats, open(os.path.join(a.out, "stats.json"), "w", encoding="utf-8", newline="\n"), indent=2)
    with open(os.path.join(a.out, "MANIFEST.txt"), "w", encoding="utf-8", newline="\n") as f:
        for k, v in sorted(manifest.items()):
            f.write(f"{v}  {k}\n")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
