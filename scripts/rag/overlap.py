#!/usr/bin/env python3
"""Is lexical overlap the reason BM25 beats the dense retrievers? (task 3.17)

  python scripts/rag/overlap.py                          # lexical half only (laptop): BM25 and overlap-only
  ~/venv-embed/bin/python scripts/rag/overlap.py --dense all --write    # VM03: + the four 3.10 embedders

Diagnostic only. It changes no adoption decision (3.10/3.11 stand) and reads only the TRAIN freeze: every train
row with a defined gold table, whole-table chunks, both query plans. Val and test are not touched.

Per row, `overlap` is the share of the query's content tokens (retrieve.tokens) that occur in the gold table.
Four measurements:

  1. stratify  recall@1 of BM25 and each dense model in three overlap strata: `zero` (no shared content token:
               BM25 cannot reach the gold at all), then the remaining rows halved by overlap into `low`/`high`
  2. overlap-only retriever  tables ranked by how many distinct query tokens they contain; no IDF, no TF.
               Ties are broken at random, reported in expectation. If it sits near BM25, overlap is the signal.
  3. de-lexicalised query  the query with every token that also occurs in the gold table deleted (stopwords
               and the rest kept). This uses the gold table, so it is a counterfactual, never a retriever: BM25
               must collapse, and the question is what a dense model keeps above chance (1/n_tables).
  4. paired    McNemar (exact) at recall@1, BM25 against each dense model, overall and per stratum, with the
               mean overlap of the rows each side wins alone.

Verdict rule, fixed after the lexical half and BEFORE any dense number (plan task 3.17), on question_only against
bge-base, where gap(s) = recall@1(BM25) - recall@1(dense) in stratum s:
  A  BM25's lead lives where overlap is high:      gap(high) > gap(low)
  B  overlap alone is enough to beat the model:    recall@1(overlap-only) >= recall@1(dense)
  supported = A and B;  partly = exactly one;  refuted = neither.
The `zero` stratum is excluded from A: BM25 scores 0 there by construction, so it would vote for the hypothesis
automatically. It is reported instead, as what dense retrieval reaches that no lexical retriever can.
chart_metadata is reported beside it, not voted: its chart-side words are the gold table's own labels, so every
row overlaps (mean 0.79, never 0) and its strata differ only in how many question words are off-table.
"""
import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from compare import mcnemar  # noqa: E402
from rag import corpus as C  # noqa: E402
from rag import metrics as M  # noqa: E402
from rag import query as Q  # noqa: E402
from rag import retrieve as R  # noqa: E402
from rag import tables as T  # noqa: E402

OUT = M.OUT / "overlap.json"
PRIMARY = ("question_only", "embed:bge-base")
BINS = ("zero", "low", "high")
WORD = re.compile(r"[A-Za-z0-9]+(?:\.[0-9]+)?")


def overlap(query, table_tokens):
    q = set(R.tokens(query))
    return round(len(q & table_tokens) / len(q), 4) if q else 0.0


def delexicalise(query, table_tokens):
    """The query with every content token that occurs in the gold table removed; other words kept in place."""
    out = WORD.sub(lambda m: "" if m.group(0).lower() in table_tokens else m.group(0), query)
    return re.sub(r"[ \t]+", " ", out).strip()


class OverlapOnly:
    """Distinct query tokens matched per table. rank() returns the gold's expected rank under random tie-breaks."""

    def __init__(self, chunks):
        self.ids = [c["chart_id"] for c in chunks]
        self.index = {}
        for i, c in enumerate(chunks):
            for t in set(R.tokens(c["text"])):
                self.index.setdefault(t, []).append(i)

    def expected_rank(self, query, gold):
        s = Counter()
        for t in set(R.tokens(query)):
            for i in self.index.get(t, ()):
                s[i] += 1
        g = s.get(self.ids.index(gold), 0)
        if g == 0:
            return None                      # unreachable: shares nothing with the query
        above = sum(v > g for v in s.values())
        tied = sum(v == g for v in s.values()) - 1
        return above + 1 + tied / 2, above, tied


def rank_of(retriever, query, gold):
    if not R.tokens(query) and isinstance(retriever, R.Lexical):
        return None                          # a lexical retriever has nothing to match
    ranked = R.rank_tables(retriever, query)
    return ranked.index(gold) + 1 if gold in ranked else None


def recall_mrr(ranks):
    n = len(ranks)
    if not n:
        return {"recall@1": None, "mrr": None, "n": 0}
    return {"recall@1": round(sum(r == 1 for r in ranks) / n, 4),
            "mrr": round(sum(1 / r for r in ranks if r) / n, 4), "n": n}


def strata(rows, plan):
    """zero overlap, then the overlapping rows halved by rank of overlap (ties split by id)."""
    zero = [r for r in rows if r["overlap"][plan] == 0]
    rest = sorted((r for r in rows if r["overlap"][plan] > 0), key=lambda r: (r["overlap"][plan], r["id"]))
    cut = {"zero": zero, "low": rest[:len(rest) // 2], "high": rest[len(rest) // 2:]}
    return {b: {r["id"] for r in part} for b, part in cut.items()}, \
        {b: [part[0]["overlap"][plan], part[-1]["overlap"][plan]] if part else None for b, part in cut.items()}


def verdict(gap, overlap_only_r1, dense_r1):
    a, b = gap["high"] > gap["low"], overlap_only_r1 >= dense_r1
    return {"A_lead_in_high_overlap": a, "B_overlap_only_beats_dense": b,
            "outcome": "supported" if a and b else "partly" if a or b else "refuted"}


def mean(xs):
    return round(sum(xs) / len(xs), 4) if xs else None


def run(dense, write):
    corpus_rows, index = T.load_corpus()
    tables_by_id = {t["chart_id"]: t for t in corpus_rows}
    chunks = C.build(corpus_rows, "whole_table")
    tok = {c["chart_id"]: set(R.tokens(c["text"])) for c in chunks}
    rows = [r for r in M.eval_rows("train", index) if r["gold"]]
    selection = {r["id"] for r in M.selection_rows(index)}
    pinned = {}
    for x in json.loads((M.OUT / "retrieval.json").read_text(encoding="utf-8"))["selection"]["results"]:
        if x["retriever"].startswith("embed:"):
            pinned[x["retriever"]] = x["model"]["revision"]

    retrievers = {"bm25": R.Lexical(chunks, "bm25")}
    models = {}
    for name in dense:
        kind = f"embed:{name}"
        retrievers[kind] = R.Embedding(chunks, name, revision=pinned.get(kind))
        models[kind] = dict(retrievers[kind].info)
        if pinned.get(kind) and models[kind]["revision"] != pinned[kind]:
            raise ValueError(f"{kind} loaded {models[kind]['revision']}, 3.10 measured {pinned[kind]}")
    ovl = OverlapOnly(chunks)

    for r in rows:
        r["overlap"], r["rank"], r["delex_rank"], r["overlap_only"] = {}, {}, {}, {}
        for plan in Q.PLANS:
            text, _ = Q.build(r["question"], plan, tables_by_id[r["image_chart_id"]])
            g = tok[r["gold"]]
            r["overlap"][plan] = overlap(text, g)
            r["overlap_only"][plan] = ovl.expected_rank(text, r["gold"])
            dl = delexicalise(text, g)
            r["rank"][plan] = {k: rank_of(v, text, r["gold"]) for k, v in retrievers.items()}
            r["delex_rank"][plan] = {k: rank_of(v, dl, r["gold"]) for k, v in retrievers.items()}
        print(f"\r{len([x for x in rows if 'rank' in x])}/{len(rows)}", end="", file=sys.stderr)
    print(file=sys.stderr)

    report = {"task": "3.17", "pool": {"split": "train", "n": len(rows), "with_selection_set": len(selection),
              "chunking": "whole_table", "tables": len(chunks)}, "corpus_sha256": index["stats"]["corpus_sha256"],
              "models": models, "chance_recall@1": round(1 / len(chunks), 5), "plans": {}}
    for plan in Q.PLANS:
        bins, ranges = strata(rows, plan)
        ov = [r["overlap"][plan] for r in rows]
        oo = [r["overlap_only"][plan] for r in rows]
        p = {"overlap": {"mean": mean(ov), "share_full": round(sum(x == 1.0 for x in ov) / len(ov), 4),
                         "share_zero": round(sum(x == 0.0 for x in ov) / len(ov), 4),
                         "strata_ranges": ranges, "strata_n": {b: len(s) for b, s in bins.items()}},
             "overlap_only": {"recall@1_expected": round(sum(1 / (1 + o[2]) for o in oo if o and o[1] == 0) / len(oo), 4),
                              "mrr_expected": round(sum(1 / o[0] for o in oo if o) / len(oo), 4),
                              "gold_tied_at_top": sum(1 for o in oo if o and o[1] == 0 and o[2] > 0),
                              "gold_unreachable": sum(1 for o in oo if o is None)},
             "retrievers": {}, "delexicalised": {}, "paired": {}, "selection_set": {}}
        for k in retrievers:
            p["retrievers"][k] = {"all": recall_mrr([r["rank"][plan][k] for r in rows]),
                                  **{b: recall_mrr([r["rank"][plan][k] for r in rows if r["id"] in bins[b]])
                                     for b in BINS}}
            p["delexicalised"][k] = recall_mrr([r["delex_rank"][plan][k] for r in rows])
            p["selection_set"][k] = recall_mrr([r["rank"][plan][k] for r in rows if r["id"] in selection])
        p["delexicalised"]["empty_queries"] = sum(
            not R.tokens(delexicalise(Q.build(r["question"], plan, tables_by_id[r["image_chart_id"]])[0],
                                      tok[r["gold"]])) for r in rows)
        for k in retrievers:
            if k == "bm25":
                continue
            top1 = {x: {r["id"]: r["rank"][plan][x] == 1 for r in rows} for x in ("bm25", k)}
            entry = {}
            for b, ids in [("all", None), *bins.items()]:
                a = {i: v for i, v in top1["bm25"].items() if ids is None or i in ids}
                d = {i: v for i, v in top1[k].items() if ids is None or i in ids}
                bm_only, dn_only, pv = mcnemar(a, d)
                entry[b] = {"bm25_only": bm_only, "dense_only": dn_only, "p": round(pv, 4),
                            "gap_recall@1": round((sum(a.values()) - sum(d.values())) / len(a), 4) if a else None, "n": len(a)}
            by_id = {r["id"]: r for r in rows}
            entry["mean_overlap_bm25_only"] = mean([by_id[i]["overlap"][plan] for i in top1["bm25"]
                                                    if top1["bm25"][i] and not top1[k][i]])
            entry["mean_overlap_dense_only"] = mean([by_id[i]["overlap"][plan] for i in top1[k]
                                                     if top1[k][i] and not top1["bm25"][i]])
            entry["verdict_rule"] = verdict({b: entry[b]["gap_recall@1"] for b in BINS},
                                            p["overlap_only"]["recall@1_expected"],
                                            p["retrievers"][k]["all"]["recall@1"])
            p["paired"][k] = entry
        report["plans"][plan] = p
    plan, k = PRIMARY
    report["verdict"] = {"plan": plan, "dense": k,
                         "rule": report["plans"][plan]["paired"].get(k, {}).get("verdict_rule", "NOT MEASURED")}
    print(json.dumps(report, indent=1))
    if write:
        OUT.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8", newline="\n")
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dense", nargs="*", default=[], choices=list(R.EMBED_MODELS) + ["all"])
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()
    run(list(R.EMBED_MODELS) if "all" in a.dense else a.dense, a.write)


if __name__ == "__main__":
    main()
