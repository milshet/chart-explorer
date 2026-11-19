#!/usr/bin/env python3
"""The table corpus and the record -> table join (task 3.0d).

  python scripts/rag/tables.py            # build in memory, check, print coverage; write nothing
  python scripts/rag/tables.py --write    # write data/v2.0/tables/{corpus.jsonl,table_index.json,MANIFEST.txt}

Every frozen record carries its chart's data table inline as `input_text`. This pulls the tables out, one per
chart, parses them to structured form and joins every record to the chart whose table answers it.

Two sources, one serialisation:
  * the frozen records (data/v2.0/image_only/{train,val,test}.jsonl, `input_text`) cover 1,084 charts;
  * the pre-labelling catalogue (data/v2.0/selection/{train,val,test}.jsonl, `chart_data_table`) covers all
    1,168 charts with an image in data/v2.0/images/, including 84 whose every question was rejected in blind
    labelling. Those 84 enter the corpus as distractors only (`provenance: "distractor_only"`).
Where the two sources share a chart they must agree byte for byte; add_distractors() raises if they ever do not.

The join is built from the frozen records alone, so a distractor-only chart can never be a gold. Unanswerable
records pair a question with an unrelated chart (chartqa_rules.make_unanswerable), so their gold table is
undefined by construction: they are joined with gold_chart_id null and reason "undefined_gold", which the
retrieval metrics (task 3.4) report as n_unscorable rather than as misses.

data/v2.0/MANIFEST.txt is not touched: the corpus has its own manifest, so no earlier run's gate moves.
"""
import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "data" / "v2.0"
OUT = DATA / "tables"
SPLITS = ("train", "val", "test")


# --- parsing ---------------------------------------------------------------------------------------

def parse(text):
    """Structured form of one serialised table: header fields, series, column names and rows of cells.

    Cells stay strings exactly as serialised ("nan" becomes None) so nothing is lost; numbers() reads values.
    """
    head, sep, body = text.partition("\nTable:\n")
    if not sep:
        raise ValueError("no 'Table:' section")
    meta = {}
    for line in head.splitlines():
        key, _, value = line.partition(": ")
        meta[key.strip()] = value.strip()
    lines = [l for l in body.splitlines() if l.strip()]
    if len(lines) < 2 or not re.fullmatch(r"\|(\s*:?-+:?\s*\|)+", lines[1].replace(" ", "")):
        raise ValueError("table has no Markdown header row")
    cells = [[c.strip() for c in l.strip().strip("|").split("|")] for l in lines]
    header, rows = cells[0], cells[2:]
    if any(len(r) != len(header) for r in rows):
        raise ValueError("ragged table")
    series = []
    for s in filter(None, (x.strip() for x in meta.get("Series", "").split(" | "))):
        m = re.fullmatch(r"(.*?)\s*\(colour: ([^;)]*)[^)]*\)", s)
        series.append({"name": m.group(1), "colour": m.group(2)} if m else {"name": s, "colour": None})
    chart_type = meta.get("Chart type", "")
    return {
        "source": meta.get("Source"),
        "chart_type": chart_type.split(",")[0].strip() or None,
        "title": meta.get("Title"),
        "category_axis": meta.get("Category axis"),
        "series": series,
        "columns": header,
        "rows": [[None if c.lower() == "nan" else c for c in r] for r in rows],
    }


def numbers(table):
    """Every numeric cell value in the table, as floats."""
    out = []
    for row in table["rows"]:
        for c in row[1:]:
            try:
                out.append(float(str(c).replace(",", "").rstrip("%")))
            except (TypeError, ValueError):
                pass
    return out


def sha256(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# --- building --------------------------------------------------------------------------------------

def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def image_split(data=DATA):
    """{chart_id: split folder of its committed image}."""
    return {p.stem: p.parent.name for p in sorted((data / "images").glob("*/*.png"))}


def from_records(data=DATA):
    """(corpus entries keyed by chart_id from frozen records, the record -> table join)."""
    tables, join = {}, {}
    for split in SPLITS:
        for r in read_jsonl(data / "image_only" / f"{split}.jsonl"):
            cid, text = r["chart_id"], r["input_text"]
            if cid in tables and tables[cid]["text"] != text:
                raise ValueError(f"{cid}: two records carry different tables")
            tables.setdefault(cid, {"chart_id": cid, "provenance": "record", "text": text})
            if r["answer_type"] == "unanswerable":
                # the question was written for question_source_chart; the image shown is chart_id
                join[r["id"]] = {"split": split, "image_chart_id": cid, "gold_chart_id": None,
                                 "reason": "undefined_gold", "question_source_chart": r["question_source_chart"]}
            else:
                join[r["id"]] = {"split": split, "image_chart_id": cid, "gold_chart_id": cid, "reason": None}
    return tables, join


def add_distractors(tables, data=DATA):
    """Add every catalogued chart that no frozen record carries, tagged distractor_only.

    Raises if the catalogue and a frozen record disagree on a chart they share: the corpus must have one
    table per chart, and a silent preference for either source would hide a broken freeze.
    """
    tables = dict(tables)
    compared = 0
    for split in SPLITS:
        for r in read_jsonl(data / "selection" / f"{split}.jsonl"):
            cid, text = r["chart_id"], r["chart_data_table"]
            if cid in tables:
                compared += 1
                if tables[cid]["text"] != text:
                    raise ValueError(f"{cid}: selection catalogue and frozen record disagree on the table")
            else:
                tables[cid] = {"chart_id": cid, "provenance": "distractor_only", "text": text}
    return tables, compared


def build(data=DATA):
    tables, join = from_records(data)
    tables, compared = add_distractors(tables, data)
    splits = image_split(data)
    missing = sorted(set(splits) - set(tables))
    if missing:
        raise ValueError(f"{len(missing)} charts have an image but no table, e.g. {missing[:3]}")
    corpus = []
    for cid in sorted(tables):
        t = tables[cid]
        corpus.append({"chart_id": cid, "image_split": splits.get(cid), "provenance": t["provenance"],
                       "sha256": sha256(t["text"]), **parse(t["text"]), "text": t["text"]})
    gold = {j["gold_chart_id"] for j in join.values() if j["gold_chart_id"]}
    distractor_only = {c["chart_id"] for c in corpus if c["provenance"] == "distractor_only"}
    if gold & distractor_only:
        raise ValueError("a distractor-only chart is some record's gold table")
    h = hashlib.sha256()
    for c in corpus:
        h.update(f"{c['chart_id']} {c['sha256']}\n".encode())
    stats = {
        "tables": len(corpus),
        "rows": sum(len(c["rows"]) for c in corpus),
        "from_records": len(corpus) - len(distractor_only),
        "distractor_only": len(distractor_only),
        "catalogue_records_compared": compared,
        "catalogue_disagreements": 0,
        "records_joined": len(join),
        "records_with_gold": sum(j["gold_chart_id"] is not None for j in join.values()),
        "records_undefined_gold": sum(j["gold_chart_id"] is None for j in join.values()),
        "corpus_sha256": h.hexdigest(),
        "by_split": {s: {"records": sum(j["split"] == s for j in join.values()),
                         "undefined_gold": sum(j["split"] == s and j["gold_chart_id"] is None for j in join.values())}
                     for s in SPLITS},
    }
    return corpus, join, stats


def write(corpus, join, stats, out=OUT):
    out.mkdir(parents=True, exist_ok=True)
    files = {
        "corpus.jsonl": "".join(json.dumps(c, ensure_ascii=False) + "\n" for c in corpus),
        "table_index.json": json.dumps({"stats": stats, "records": join}, indent=1, ensure_ascii=False, sort_keys=True) + "\n",
    }
    for name, text in files.items():
        (out / name).write_text(text, encoding="utf-8", newline="\n")
    manifest = "".join(f"{sha256(files[n])}  {n}\n" for n in sorted(files))
    (out / "MANIFEST.txt").write_text(manifest, encoding="utf-8", newline="\n")


def load_corpus(out=OUT):
    """The committed corpus, checked against its manifest. Raises on a mismatch."""
    manifest = dict(reversed(l.split()) for l in (out / "MANIFEST.txt").read_text(encoding="utf-8").splitlines() if l)
    for name, digest in manifest.items():
        if hashlib.sha256((out / name).read_bytes()).hexdigest() != digest:
            raise ValueError(f"{out / name} does not match its MANIFEST.txt")
    corpus = read_jsonl(out / "corpus.jsonl")
    index = json.loads((out / "table_index.json").read_text(encoding="utf-8"))
    return corpus, index


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()
    corpus, join, stats = build()
    print(json.dumps(stats, indent=1))
    if args.write:
        write(corpus, join, stats)
        print(f"wrote {OUT.relative_to(REPO)}/")


if __name__ == "__main__":
    sys.exit(main())
