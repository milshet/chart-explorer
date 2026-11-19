#!/usr/bin/env python3
"""Frozen RAG prompt sets for tasks 3.5 and 3.6.

  python scripts/rag/prompts.py            # build, print retrieval accuracy per set; write nothing
  python scripts/rag/prompts.py --write    # write data/rag/ and its MANIFEST.txt

Each set is the image_table prompt format (image + table as text + question), with the table the RETRIEVER
returned (top-1, the configuration chosen on the selection set in results/rag/retrieval.json) in place of the
gold one. Ids are the source split's ids, so every condition is paired item for item with Eval-A.

  rag_val            val, chart_metadata queries                (checkpoint/condition checks)
  rag_test           test, chart_metadata queries               (3.5 headline: the perfect-OCR end of the bracket)
  rag_test_qo        test, question_only queries                (3.5: the floor end of the bracket)
  rag_test_withheld  test, chart_metadata, gold table WITHHELD from the corpus for half the answerable rows
                     (seed 51003): retrieval returns a confidently ranked wrong table there (task 3.6)

The table is introduced as possibly not matching the chart, in every set, so the 3.5 and 3.6 conditions see the
same preamble and only the retrieval differs. Records carry retrieved_chart_id, retrieved_table,
retrieval_correct and withheld; input_text stays the shown chart's own table, for evidence grounding.
"""
import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rag import corpus as C  # noqa: E402
from rag import query as Q  # noqa: E402
from rag import retrieve as R  # noqa: E402
from rag import tables as T  # noqa: E402

OUT = T.REPO / "data" / "rag"
PREAMBLE = "Chart data (retrieved from a table store; it may not correspond to this chart):\n"
WITHHOLD_SEED = 51003
SETS = {"rag_val": ("val", "chart_metadata", False), "rag_test": ("test", "chart_metadata", False),
        "rag_test_qo": ("test", "question_only", False), "rag_test_withheld": ("test", "chart_metadata", True)}


def chosen():
    r = json.loads((T.REPO / "results/rag/retrieval.json").read_text(encoding="utf-8"))["chosen"]
    return r["chunking"], r["retriever"]


def with_table(record, table_text):
    """The image_table record with its table replaced and introduced by the retrieval preamble."""
    user = record["messages"][1]["content"]
    text = user[1]["text"]
    if not text.startswith("Chart data:\n"):
        raise ValueError(f"{record['id']}: not an image_table prompt")
    question_part = text[text.index("\n\nQuestion: ") + 2:]
    new_text = f"{PREAMBLE}{table_text}\n\n{question_part}"
    image = f"../v2.0/{record['image']}"  # the set lives in data/rag; images stay in data/v2.0
    msgs = [record["messages"][0],
            {"role": "user", "content": [{**user[0], "image": image}, {"type": "text", "text": new_text}]},
            record["messages"][2]]
    # input_text stays the shown chart's own table: the scorer's evidence grounding reads it, and grounding must
    # mean the same thing as on Eval-A. The retrieved table is in the prompt and in retrieved_table.
    return {**record, "image": image, "retrieved_table": table_text, "messages": msgs}


def build():
    corpus, index = T.load_corpus()
    by_id = {t["chart_id"]: t for t in corpus}
    chunking, kind = chosen()
    chunks = C.build(corpus, chunking)
    full = R.make(chunks, kind)
    out, stats = {}, {}
    for name, (split, plan, withhold) in SETS.items():
        withheld = set()
        if withhold:
            answerable = sorted(r["id"] for r in T.read_jsonl(T.DATA / "image_only" / f"{split}.jsonl")
                                if index["records"][r["id"]]["gold_chart_id"])
            withheld = set(random.Random(WITHHOLD_SEED).sample(answerable, len(answerable) // 2))
        files = {}
        for targets, sub in (("reasoning", ""), ("answer_only", "answer_only/")):
            rows = []
            for rec in T.read_jsonl(T.DATA / "image_table" / f"{sub}{split}.jsonl"):
                j = index["records"][rec["id"]]
                text, _ = Q.build(rec["question"], plan, by_id[j["image_chart_id"]])
                ranked = R.rank_tables(full, text, k=6)
                if rec["id"] in withheld:
                    ranked = [c for c in ranked if c != j["gold_chart_id"]]
                top = ranked[0]
                rows.append({**with_table(rec, by_id[top]["text"]), "split": name, "rag_source_split": split,
                             "query_plan": plan, "retrieved_chart_id": top,
                             "retrieval_correct": None if j["gold_chart_id"] is None else top == j["gold_chart_id"],
                             "withheld": rec["id"] in withheld})
            files[f"image_table/{sub}{name}.jsonl"] = rows
        rows = files[f"image_table/{name}.jsonl"]
        scorable = [r for r in rows if r["retrieval_correct"] is not None]
        stats[name] = {"source_split": split, "query_plan": plan, "n": len(rows), "n_scorable": len(scorable),
                       "top1_correct": sum(r["retrieval_correct"] for r in scorable), "withheld": len(withheld)}
        out.update(files)
    return out, {"chunking": chunking, "retriever": kind, "withhold_seed": WITHHOLD_SEED,
                 "corpus_sha256": index["stats"]["corpus_sha256"], "sets": stats}


def write(files, stats):
    text = {rel: "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows) for rel, rows in files.items()}
    for rel, t in text.items():
        (OUT / rel).parent.mkdir(parents=True, exist_ok=True)
        (OUT / rel).write_text(t, encoding="utf-8", newline="\n")
    (OUT / "stats.json").write_text(json.dumps(stats, indent=1) + "\n", encoding="utf-8", newline="\n")
    (OUT / "MANIFEST.txt").write_text("".join(f"{hashlib.sha256(text[r].encode()).hexdigest()}  {r}\n"
                                              for r in sorted(text)), encoding="utf-8", newline="\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()
    files, stats = build()
    print(json.dumps(stats, indent=1))
    if args.write:
        write(files, stats)


if __name__ == "__main__":
    main()
