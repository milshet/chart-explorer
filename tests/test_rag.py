"""CPU-only checks for the Week 3 retrieval work: table corpus (3.0d), queries, retrieval and metrics."""
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
from rag import tables  # noqa: E402

SAMPLE = """Source: OECD
Chart type: line, 2 series
Title: Investment by sector
Category axis: Country
Series: Australia (colour: blue) | Chile (colour: red; single series, all bars share this colour)
Table:
| Country | 1975 | 1980 |
|---|---|---|
| Chile | nan | 24.5 |
| Australia | 38.0 | 1,037 |"""


# --- table corpus (task 3.0d) ----------------------------------------------------------------------

def test_parse_keeps_cells_and_reads_header_fields():
    t = tables.parse(SAMPLE)
    assert (t["source"], t["chart_type"], t["title"], t["category_axis"]) == ("OECD", "line", "Investment by sector", "Country")
    assert t["series"] == [{"name": "Australia", "colour": "blue"}, {"name": "Chile", "colour": "red"}]
    assert t["columns"] == ["Country", "1975", "1980"]
    assert t["rows"] == [["Chile", None, "24.5"], ["Australia", "38.0", "1,037"]]
    assert tables.numbers(t) == [24.5, 38.0, 1037.0]


@pytest.mark.parametrize("bad", ["Source: x\n| a |\n|---|", SAMPLE.replace("|---|---|---|", "| x | y | z |"),
                                 SAMPLE + "\n| extra |"])
def test_parse_rejects_malformed_tables(bad):
    with pytest.raises(ValueError):
        tables.parse(bad)


@pytest.fixture(scope="module")
def committed():
    return tables.load_corpus()


def test_committed_corpus_is_what_the_builder_produces(committed):
    corpus, index = committed
    built, join, stats = tables.build()
    assert corpus == built and index == {"stats": stats, "records": join}


def test_committed_corpus_covers_every_chart_in_the_freeze(committed):
    corpus, index = committed
    ids = {c["chart_id"] for c in corpus}
    assert ids == set(tables.image_split()) and len(ids) == len(corpus) == 1168
    distractors = {c["chart_id"] for c in corpus if c["provenance"] == "distractor_only"}
    golds = {r["gold_chart_id"] for r in index["records"].values() if r["gold_chart_id"]}
    assert len(distractors) == 84 and not (distractors & golds)


def test_every_record_is_joined_and_unanswerable_gold_is_undefined(committed):
    _, index = committed
    for split in tables.SPLITS:
        for r in tables.read_jsonl(tables.DATA / "image_only" / f"{split}.jsonl"):
            j = index["records"][r["id"]]
            if r["answer_type"] == "unanswerable":
                assert j["gold_chart_id"] is None and j["reason"] == "undefined_gold"
            else:
                assert j["gold_chart_id"] == r["chart_id"]
    assert index["stats"]["records_undefined_gold"] == 57


def test_distractors_raise_when_the_two_sources_disagree():
    recs, _ = tables.from_records()
    cid = next(iter(recs))
    with pytest.raises(ValueError):
        tables.add_distractors({**recs, cid: {**recs[cid], "text": recs[cid]["text"] + " "}})


def test_corpus_manifest_detects_an_edit(tmp_path, committed):
    corpus, index = committed
    tables.write(corpus, index["records"], index["stats"], tmp_path)
    tables.load_corpus(tmp_path)
    with open(tmp_path / "corpus.jsonl", "a", encoding="utf-8") as f:
        f.write("\n")
    with pytest.raises(ValueError):
        tables.load_corpus(tmp_path)
