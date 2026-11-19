#!/usr/bin/env python3
"""Retrieval metrics and the selection experiment (task 3.4).

  python scripts/rag/metrics.py                 # selection set: every chunking x plan x lexical retriever
  python scripts/rag/metrics.py --write         # also score the chosen config on val and test; write results/rag/
  python scripts/rag/metrics.py --retrievers bm25 tfidf embed --write   # on VM03, with sentence-transformers

recall@1/3/5 and MRR, separately from answer quality. Rows whose gold table is undefined (the ~5% unanswerable
items pair a question with an unrelated chart) are excluded from the denominator and reported as n_unscorable:
counting them as misses would understate a retriever that works on everything it can reach.

Chunking, retriever and query plan are choices made by looking at a number, so they are made on a SELECTION SET
drawn from the training freeze (SELECTION_N rows, seed 41003), never on the reported split: val selects / test
reports, one level out from checkpoints. Both figures are written, the selection one (what decided) and the
headline one (what the report quotes, with its n). If every configuration scores 1.0 at some k, the metric is
saturated there and the choice is made on MRR, which the output says.
"""
import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rag import corpus as C  # noqa: E402
from rag import query as Q  # noqa: E402
from rag import retrieve as R  # noqa: E402
from rag import tables as T  # noqa: E402

REPO = T.REPO
OUT = REPO / "results" / "rag"
KS = (1, 3, 5)
SELECTION_N, SELECTION_SEED = 120, 41003


def eval_rows(split, index):
    """Rows of a split joined to their gold table: [{"id", "question", "image_chart_id", "gold"}]."""
    rows = []
    for r in T.read_jsonl(T.DATA / "image_only" / f"{split}.jsonl"):
        j = index["records"][r["id"]]
        rows.append({"id": r["id"], "question": r["question"], "image_chart_id": j["image_chart_id"],
                     "gold": j["gold_chart_id"]})
    return rows


def selection_rows(index):
    rows = [r for r in eval_rows("train", index) if r["gold"]]
    return sorted(random.Random(SELECTION_SEED).sample(rows, SELECTION_N), key=lambda r: r["id"])


def score(rows, retriever, plan, tables_by_id):
    ranks, leaks, per_row = [], 0, []
    for r in rows:
        # the query is built from the chart the user sees (its labels), which for an unanswerable row is the
        # unrelated chart shown, not the question's own
        text, leak = Q.build(r["question"], plan, tables_by_id[r["image_chart_id"]])
        leaks += leak
        if r["gold"] is None:
            per_row.append({"id": r["id"], "rank": None})
            continue
        ranked = R.rank_tables(retriever, text)
        rank = ranked.index(r["gold"]) + 1 if r["gold"] in ranked else None
        ranks.append(rank)
        per_row.append({"id": r["id"], "rank": rank, "top": ranked[:5]})
    n = len(ranks)
    out = {f"recall@{k}": round(sum(1 for x in ranks if x and x <= k) / n, 4) for k in KS}
    out.update({"mrr": round(sum(1 / x for x in ranks if x) / n, 4), "n_scorable": n,
                "n_unscorable": len(rows) - n, "question_value_leaks": leaks})
    return out, per_row


def configs(retrievers):
    return [(s, p, r) for s in C.STRATEGIES for p in Q.PLANS for r in retrievers]


def choose(results):
    """The best config on the selection set: recall@1, then MRR. Says when recall@1 is saturated."""
    best = max(results, key=lambda x: (x["metrics"]["recall@1"], x["metrics"]["mrr"]))
    saturated = sum(x["metrics"]["recall@1"] == 1.0 for x in results) > 1
    return best, ("mrr (recall@1 saturated)" if saturated else "recall@1, then mrr")


def run(retrievers, write):
    corpus_rows, index = T.load_corpus()
    tables_by_id = {t["chart_id"]: t for t in corpus_rows}
    sel = selection_rows(index)
    built = {s: C.build(corpus_rows, s) for s in C.STRATEGIES}
    cache, results = {}, []
    for strategy, plan, kind in configs(retrievers):
        if (strategy, kind) not in cache:
            cache[(strategy, kind)] = R.make(built[strategy], kind)
        m, _ = score(sel, cache[(strategy, kind)], plan, tables_by_id)
        results.append({"chunking": strategy, "plan": plan, "retriever": kind, "metrics": m})
        print(f"selection  {strategy:12} {plan:15} {kind:6} " + "  ".join(f"{k} {v}" for k, v in m.items()))
    best, basis = choose(results)
    print(f"chosen on the selection set ({basis}): {best['chunking']} / {best['plan']} / {best['retriever']}")
    report = {"corpus_sha256": index["stats"]["corpus_sha256"], "selection": {"n": len(sel), "seed": SELECTION_SEED,
              "from": "train", "results": results}, "chosen": {k: best[k] for k in ("chunking", "plan", "retriever")},
              "chosen_basis": basis, "headline": {}}
    retr = cache[(best["chunking"], best["retriever"])]
    for split in ("val", "test"):
        for plan in Q.PLANS:   # the bracket: both plans on the chosen chunking and retriever
            m, per_row = score(eval_rows(split, index), retr, plan, tables_by_id)
            report["headline"][f"{split}/{plan}"] = m
            print(f"{split:9}  {plan:15} " + "  ".join(f"{k} {v}" for k, v in m.items()))
            if write:
                OUT.mkdir(parents=True, exist_ok=True)
                with open(OUT / f"ranks_{split}_{plan}.jsonl", "w", encoding="utf-8", newline="\n") as f:
                    f.writelines(json.dumps(r) + "\n" for r in per_row)
    if write:
        (OUT / "retrieval.json").write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8", newline="\n")
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--retrievers", nargs="+", default=["tfidf", "bm25"], choices=R.RETRIEVERS)
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()
    run(args.retrievers, args.write)


if __name__ == "__main__":
    main()
