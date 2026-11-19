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


# --- red-team charts (task 3.0g) -------------------------------------------------------------------

@pytest.fixture(scope="module")
def redteam():
    import redteam_charts
    return redteam_charts, redteam_charts.build()


def test_redteam_files_regenerate_byte_identical(redteam):
    import hashlib
    rc, built = redteam
    manifest = dict(reversed(l.split()) for l in (rc.OUT / "MANIFEST.txt").read_text().splitlines() if l)
    for s, png, rows in built:
        assert hashlib.sha256(png).hexdigest() == manifest[s["image"]], s["id"]
        assert hashlib.sha256(rc.csv_text(rows).encode()).hexdigest() == manifest[s["table_csv"]]
    specs = "".join(json.dumps(s, ensure_ascii=False) + "\n" for s, _, _ in built).encode()
    assert hashlib.sha256(specs).hexdigest() == manifest["specs.jsonl"]


def test_redteam_families_split_three_abstain_four_answer(redteam):
    rc, built = redteam
    from collections import Counter
    assert Counter(rc.FAMILIES.values()) == {"answer": 4, "abstain": 3} and len(built) == 21
    for s, _, _ in built:
        assert (s["answer"] is None) == (s["expected_behaviour"] == "abstain")
        assert s["answer"] not in s["misled_answers"]           # a misled answer must be distinguishable


def test_redteam_ground_truth_is_in_its_own_table(redteam):
    _, built = redteam
    for s, _, rows in built:
        t = tables.parse(s["input_text"])
        assert t["columns"] == [str(c) for c in rows[0]]
        assert t["rows"] == [[str(c) for c in r] for r in rows[1:]]
        if s["expected_behaviour"] == "abstain":
            assert not any(e in s["input_text"].split("Table:")[1] for e in s["absent_entities"]), s["id"]
        if s["caption"]:
            assert str(s["misled_answers"][0]) not in s["input_text"]   # the caption's figure is not plotted


def test_redteam_seeds_and_payloads_are_disjoint_from_the_dataset(redteam):
    _, built = redteam
    seeds = [s["seed"] for s, _, _ in built]
    assert len(set(seeds)) == 21 and min(seeds) >= 700_000 and not {2026, 31003} & set(seeds)
    tokens = [s["injection_token"] for s, _, _ in built if s["injection_token"]]
    assert len(set(tokens)) == 3
    corpus = (REPO / "data/v2.0/tables/corpus.jsonl").read_text(encoding="utf-8")
    assert not any(tok in corpus for tok in tokens)
