#!/usr/bin/env python3
"""Build Eval-C, the abstention probe set (task 3.0c).

  python scripts/build_eval_c.py            # build in memory, print the balance, write nothing
  python scripts/build_eval_c.py --write    # write data/eval_c/ and its MANIFEST.txt

Eval-A has nine unanswerable items and every fine-tuned run scores 9/9 on them: a saturated metric. Eval-C is
~50 probes, each unanswerable for a reason that is *proved at construction* from the data tables, never assumed:

  series_absent      a real unused human question names a series (a column header of its own chart's table)
                     and a category (a row label); the paired chart's table carries that category but not the series
  category_absent    the same, the other way round: the paired chart has the series, not the category
  not_in_chart       no table entity the question names occurs anywhere in the paired chart's table
  requires_external_knowledge
                     the question names a row and a series that ARE in the chart, and asks for a quantity the
                     table cannot supply (an inflation adjustment, a world share, a cause, a currency conversion);
                     no catalogue question can be proved to need outside data, so these are templated and marked
                     origin "template". The keyword the quantity turns on is checked absent from the table.

Questions come from the finance/economy questions in data/v2.0/selection/all_domains_catalogue.jsonl that
selection did not use, restricted to charts whose table is in the corpus (data/v2.0/tables, task 3.0d) so that
"the question names this entity" is checked against the question's own table. Paired charts are val/test charts
(held out from training; §9 decision 10), so Eval-C shares images with Eval-A: the same chart is answerable there
and unanswerable here, which is what tests that the model refuses on the question rather than the chart's look.

Records have exactly the shape of data/v2.0 records, in the same four variant/target files, so evaluate.py scores
them with --split eval_c. Image paths point into data/v2.0/images. data/v2.0/MANIFEST.txt is not touched; Eval-C
has its own. Seed 31003 (selection used 2026; the red-team generator uses a third range). Never trained on.
"""
import argparse
import hashlib
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
from rag import tables  # noqa: E402

V2 = REPO / "data" / "v2.0"
OUT = REPO / "data" / "eval_c"
SEED = 31003
QUOTA = {"series_absent": 13, "category_absent": 13, "not_in_chart": 12, "requires_external_knowledge": 12}
FILES = {("image_only", "reasoning"): "image_only/eval_c.jsonl",
         ("image_only", "answer_only"): "image_only/answer_only/eval_c.jsonl",
         ("image_table", "reasoning"): "image_table/eval_c.jsonl",
         ("image_table", "answer_only"): "image_table/answer_only/eval_c.jsonl"}
GOLD = {"reasoning": {"evidence": [], "rationale": None, "answer_type": "unanswerable", "unit": None, "answer": None},
        "answer_only": {"answer_type": "unanswerable", "unit": None, "answer": None}}
# requires_external_knowledge: (template, keyword that must be absent from the table)
EXTERNAL = [
    ("For {category}, {series}: what is the value after adjusting for inflation to 2023 prices?", "inflation"),
    ("For {category}, {series}: what share of the world total does this value represent?", "world"),
    ("For {category}, {series}: what caused the value to be what it is?", "caus"),
    ("For {category}, {series}: what is the value when converted into Japanese yen?", "yen"),
]


def norm(s):
    return re.sub(r"\s+", " ", str(s).lower()).strip()


def occurs(entity, text):
    """Whole-word, case-insensitive occurrence of entity in text."""
    e = norm(entity)
    return len(e) >= 2 and re.search(r"(?<!\w)" + re.escape(e) + r"(?!\w)", norm(text)) is not None


GENERIC = {"value", "values", "characteristic", "total", "share of respondents", "percentage of respondents",
           "number of respondents", "percentage", "share", "number"}


def is_numeric(s):
    return re.fullmatch(r"[\d.,%\s/-]+", s) is not None


def entities(table):
    """(row labels = categories, column headers = series) of a parsed table.

    A column header only counts as a series when the chart has two or more data columns and the header is not
    a generic measure name: in a one-series chart the header names the measure ("Revenue in million U.S.
    dollars"), and "Value" names nothing, so its absence elsewhere proves nothing about the question.
    """
    cols = table["columns"][1:]
    series = {c for c in cols if norm(c) not in GENERIC} if len(cols) >= 2 else set()
    return ({r[0] for r in table["rows"] if r[0] and norm(r[0]) not in GENERIC}, series)


def prompt_parts():
    """System prompt and schema suffix per (variant, targets), read from the frozen test files."""
    parts = {}
    for (variant, targets), rel in FILES.items():
        ref = json.loads(open(V2 / rel.replace("eval_c", "test"), encoding="utf-8").readline())
        text = ref["messages"][1]["content"][1]["text"]
        parts[(variant, targets)] = (ref["messages"][0]["content"], text[text.index("\n\nRespond with JSON"):])
    return parts


def messages(parts, variant, targets, image, question, table_text):
    """The conversation exactly as data/v2.0 builds it (checked against every frozen test record in tests)."""
    system, suffix = parts[(variant, targets)]
    text = (f"Chart data:\n{table_text}\n\n" if variant == "image_table" else "") + f"Question: {question}{suffix}"
    return [{"role": "system", "content": system},
            {"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": text}]},
            {"role": "assistant", "content": json.dumps(GOLD[targets])}]


def candidates(corpus):
    """Unused finance/economy catalogue questions on corpus charts, with the entities they name."""
    # a selected question can reappear under another id (make_unanswerable appends "-unans" and moves it to an
    # unrelated chart), so exclude by question text as well as by id
    selected = [r for s in tables.SPLITS for r in tables.read_jsonl(V2 / "selection" / f"{s}.jsonl")]
    used_ids = {r["id"].removesuffix("-unans") for r in selected}
    used_text = {norm(r["question"]) for r in selected}
    out = []
    for q in tables.read_jsonl(V2 / "selection" / "all_domains_catalogue.jsonl"):
        if not q["in_finance_economy"] or q["id"] in used_ids or norm(q["question"]) in used_text                 or q["chart_id"] not in corpus:
            continue
        rows, cols = entities(corpus[q["chart_id"]])
        out.append({**q, "rows": sorted(r for r in rows if occurs(r, q["question"])),
                    "cols": sorted(c for c in cols if occurs(c, q["question"]))})
    return out


def prove(reason, q, chart):
    """The absent entity if pairing question q with chart proves `reason`, else None."""
    text = chart["text"]
    if reason == "series_absent" and q["rows"] and q["cols"]:
        if all(occurs(r, text) for r in q["rows"]) and not any(occurs(c, text) for c in q["cols"]):
            return {"absent": q["cols"], "present": q["rows"]}
    if reason == "category_absent" and q["rows"] and q["cols"]:
        if all(occurs(c, text) for c in q["cols"]) and not any(occurs(r, text) for r in q["rows"]):
            return {"absent": q["rows"], "present": q["cols"]}
    if reason == "not_in_chart" and any(not is_numeric(e) for e in q["rows"] + q["cols"]):
        if not any(occurs(e, text) for e in q["rows"] + q["cols"]):
            return {"absent": q["rows"] + q["cols"], "present": []}
    return None


def external_probe(chart, rng):
    """A templated requires_external_knowledge question over chart's own entities, or None."""
    rows, cols = entities(chart)
    rows = sorted(r for r in rows if len(r) <= 40)
    cols = sorted(c for c in cols if len(c) <= 60)
    if not rows or not cols:
        return None
    category, series = rng.choice(rows), rng.choice(cols)
    ratio = re.search(r"%|percent|share|rate|ratio", series, re.I) is not None
    # inflation adjustment and currency conversion are meaningless for a ratio; the cause and world share are not
    usable = [(t, k) for t, k in EXTERNAL if k not in norm(chart["text"])
              and not (ratio and k in ("inflation", "yen", "world"))]
    if not usable:
        return None
    template, keyword = rng.choice(usable)
    return {"question": template.format(series=series, category=category), "keyword": keyword,
            "present": [category, series], "absent": [keyword]}


def build():
    corpus = {c["chart_id"]: c for c in tables.load_corpus()[0]}
    held = sorted(cid for cid, c in corpus.items() if c["image_split"] in ("val", "test") and c["provenance"] == "record")
    rng = random.Random(SEED)
    pool = candidates(corpus)
    rng.shuffle(pool)
    probes, used_q, chart_use = [], set(), Counter()
    for reason, quota in QUOTA.items():
        got = 0
        if reason == "requires_external_knowledge":
            for cid in rng.sample(held, len(held)):
                if got == quota:
                    break
                if chart_use[cid]:
                    continue
                p = external_probe(corpus[cid], rng)
                if p:
                    probes.append({"reason": reason, "chart_id": cid, "origin": "template", "source_question": None,
                                   "question_source_chart": cid, **p})
                    chart_use[cid] += 1
                    got += 1
        else:
            for q in pool:
                if got == quota:
                    break
                if q["id"] in used_q:
                    continue
                # least-used held-out chart first, ties broken by the seeded order
                order = sorted((c for c in rng.sample(held, len(held)) if c != q["chart_id"]), key=lambda c: chart_use[c])
                for cid in order:
                    proof = prove(reason, q, corpus[cid])
                    if proof:
                        probes.append({"reason": reason, "chart_id": cid, "origin": "catalogue",
                                       "source_question": q["id"], "question_source_chart": q["chart_id"],
                                       "question": q["question"], **proof})
                        used_q.add(q["id"])
                        chart_use[cid] += 1
                        got += 1
                        break
        if got < quota:
            raise SystemExit(f"{reason}: only {got} of {quota} probes could be proved")
    return probes, corpus


def records(probes, corpus):
    parts = prompt_parts()
    out = {k: [] for k in FILES}
    for i, p in enumerate(probes):
        chart = corpus[p["chart_id"]]
        image = f"../v2.0/images/{chart['image_split']}/{p['chart_id']}.png"
        base = {"id": f"eval_c-{i:03d}-{p['reason']}", "split": "eval_c", "chart_id": p["chart_id"],
                "question_source_chart": p["question_source_chart"], "source": chart["source"].lower() if chart["source"] else None,
                "chart_type": chart["chart_type"], "question": p["question"], "gold_answer_raw": None, "answer": None,
                "answer_type": "unanswerable", "question_type": "unanswerable", "abstain_reason": p["reason"],
                "absent_entities": p["absent"], "present_entities": p["present"], "origin": p["origin"],
                "source_question_id": p["source_question"], "image": image, "input_text": chart["text"],
                "version": "eval_c-1.0", "confidence": None}
        for (variant, targets) in FILES:
            out[(variant, targets)].append({**base, "variant": variant,
                                            "messages": messages(parts, variant, targets, image, p["question"], chart["text"])})
    return out


def write(recs, probes):
    text = {}
    for key, rel in FILES.items():
        text[rel] = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in recs[key])
    for rel, t in text.items():
        (OUT / rel).parent.mkdir(parents=True, exist_ok=True)
        (OUT / rel).write_text(t, encoding="utf-8", newline="\n")
    stats = {"n": len(probes), "seed": SEED, "by_reason": dict(Counter(p["reason"] for p in probes)),
             "by_origin": dict(Counter(p["origin"] for p in probes)),
             "distinct_charts": len({p["chart_id"] for p in probes}),
             "corpus_sha256": tables.load_corpus()[1]["stats"]["corpus_sha256"]}
    (OUT / "stats.json").write_text(json.dumps(stats, indent=1) + "\n", encoding="utf-8", newline="\n")
    manifest = "".join(f"{hashlib.sha256((OUT / rel).read_bytes()).hexdigest()}  {rel}\n" for rel in sorted(text))
    (OUT / "MANIFEST.txt").write_text(manifest, encoding="utf-8", newline="\n")
    return stats


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()
    probes, corpus = build()
    recs = records(probes, corpus)
    for p in probes[::6]:
        print(f"{p['reason']:28} {p['chart_id'][:30]:30} absent={p['absent']}  Q: {p['question'][:90]}")
    print(Counter(p["reason"] for p in probes), "charts:", len({p["chart_id"] for p in probes}))
    if args.write:
        print(write(recs, probes))


if __name__ == "__main__":
    main()
