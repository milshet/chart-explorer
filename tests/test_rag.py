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


# --- retrieval: chunking, queries, retrievers, metrics (tasks 3.1-3.4) -----------------------------

from rag import corpus as rcorpus  # noqa: E402
from rag import metrics as rmetrics  # noqa: E402
from rag import query as rquery  # noqa: E402
from rag import retrieve as rretrieve  # noqa: E402


def test_every_row_group_chunk_repeats_the_header():
    t = tables.parse(SAMPLE.replace("| Australia | 38.0 | 1,037 |", "\n".join(f"| R{i} | {i} | {i} |" for i in range(12))))
    chunks = rcorpus.chunks({**t, "chart_id": "c"}, "row_group")
    assert len(chunks) == 3 and all(c["text"].startswith(rcorpus.header(t)) for c in chunks)
    assert len(rcorpus.chunks({**t, "chart_id": "c"}, "whole_table")) == 1


def test_query_gate_raises_on_a_data_value_and_exempts_labels():
    t = tables.parse(SAMPLE)
    q, leak = rquery.build("What was Chile in 1980?", "chart_metadata", t)   # 1980 is a column label: exempt
    assert "Investment by sector" in q and not leak
    with pytest.raises(rquery.LeakError):
        rquery.gate("Chile 24.5", t)                                          # 24.5 is a data cell
    assert rquery.build("Is Chile 24.5 in 1980?", "question_only", t)[1]     # a question leak is counted, not raised


def test_chunks_collapse_to_tables_before_k():
    chunks = [{"chart_id": "a", "chunk_id": "a#0", "text": "revenue emea"},
              {"chart_id": "a", "chunk_id": "a#1", "text": "revenue emea q3"},
              {"chart_id": "b", "chunk_id": "b#0", "text": "revenue apac"}]
    r = rretrieve.make(chunks, "bm25")
    assert rretrieve.rank_tables(r, "emea revenue q3", k=2) == ["a", "b"]   # two chunks of a count once


def test_unanswerable_rows_are_unscorable_not_misses():
    corpus, index = tables.load_corpus()
    rows = rmetrics.eval_rows("test", index)
    by_id = {t["chart_id"]: t for t in corpus}
    r = rretrieve.make(rcorpus.build(corpus, "whole_table"), "bm25")
    m, per_row = rmetrics.score(rows, r, "question_only", by_id)
    assert (m["n_scorable"], m["n_unscorable"]) == (171, 9)
    assert sum(p["rank"] is None and "top" not in p for p in per_row) == 9


def test_selection_set_is_drawn_from_train_only():
    _, index = tables.load_corpus()
    sel = rmetrics.selection_rows(index)
    assert len(sel) == rmetrics.SELECTION_N >= 80 and all(r["id"].startswith("train-") for r in sel)


def test_committed_retrieval_results_reproduce():
    committed = json.loads((REPO / "results/rag/retrieval.json").read_text(encoding="utf-8"))
    assert rmetrics.run(["tfidf", "bm25"], write=False) == committed


# --- RAG prompt sets (tasks 3.5, 3.6) --------------------------------------------------------------

def rag_rows(name, sub=""):
    return [json.loads(l) for l in open(REPO / f"data/rag/image_table/{sub}{name}.jsonl", encoding="utf-8")]


def test_rag_prompt_sets_rebuild_identically():
    import hashlib
    from rag import prompts
    files, _ = prompts.build()
    manifest = dict(reversed(l.split()) for l in (REPO / "data/rag/MANIFEST.txt").read_text().splitlines() if l)
    assert set(manifest) == set(files)
    for rel, rows in files.items():
        text = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
        assert hashlib.sha256(text.encode()).hexdigest() == manifest[rel], rel


def test_rag_sets_pair_item_for_item_with_eval_a_and_only_the_table_differs():
    from rag import prompts
    test ={json.loads(l)["id"]: json.loads(l) for l in open(REPO / "data/v2.0/image_table/test.jsonl", encoding="utf-8")}
    for name in ("rag_test", "rag_test_qo", "rag_test_withheld"):
        rows = rag_rows(name)
        assert [r["id"] for r in rows] == list(test)
        for r in rows:
            src = test[r["id"]]
            assert r["input_text"] == src["input_text"]                           # grounding still vs the shown chart
            assert r["messages"][2] == src["messages"][2] and r["messages"][0] == src["messages"][0]
            text = r["messages"][1]["content"][1]["text"]
            question = src["messages"][1]["content"][1]["text"].split("\n\nQuestion: ", 1)[1]
            assert text.startswith(prompts.PREAMBLE + r["retrieved_table"]) and text.endswith("Question: " + question)
            assert r["image"] == "../v2.0/" + src["image"]


def test_withheld_rows_never_get_their_gold_table():
    _, index = tables.load_corpus()
    rows = rag_rows("rag_test_withheld")
    held = [r for r in rows if r["withheld"]]
    assert len(held) == 85 and all(r["retrieved_chart_id"] != index["records"][r["id"]]["gold_chart_id"] for r in held)


def test_rag_sets_resolve_in_evaluate(tmp_path):
    import evaluate
    d = tmp_path / "run_x" / "adapter"
    d.mkdir(parents=True)
    (d / "adapter_model.safetensors").write_bytes(b"w")
    (d / "train_config.json").write_text(json.dumps({"run_name": "run_x", "data_dir": "data/v2.0", "variant": "image_only",
                                                     "targets": "reasoning", "max_pixels": 800 * 800}))
    with pytest.raises(SystemExit):     # rag_test is a reported set
        evaluate.resolve(evaluate.parse_args(["--split", "rag_test", "--adapter", str(d)]))
    argv = ["--split", "rag_test", "--adapter", str(d), "--final-test"]
    s = evaluate.resolve(evaluate.parse_args(argv))
    assert s["variant"] == "image_table" and s["prompt_format_new_to_model"] and s["items_dir"] == "data/rag"
    items, _, fp = evaluate.load_rows(s, "rag_test", evaluate.parse_args(argv))
    assert len(items) == 180 and fp["data_dir"] == "data/rag"
    s = evaluate.resolve(evaluate.parse_args(["--split", "rag_val", "--adapter", str(d)]))   # val-side: no --final-test
    assert s["prompt_format_new_to_model"]


# --- three-way comparison and retrieval-failure verdict (tasks 3.5, 3.6) ---------------------------

def preds(correct, abstained=None, una=()):
    return {i: {"id": i, "correct": c, "abstained": (abstained or {}).get(i, False),
                "answer_type": "unanswerable" if i in una else "number"} for i, c in correct.items()}


def test_threeway_marks_missing_conditions_and_pairs_the_rest():
    import threeway
    ids = [f"q{i}" for i in range(10)]
    data = {"b": preds({i: n < 4 for n, i in enumerate(ids)}), "f": preds({i: n < 8 for n, i in enumerate(ids)})}
    out = threeway.comparison({"base": "b", "fine-tuned": "f", "base+RAG": "missing"}, loader=data.get)
    assert out["conditions"]["base+RAG"] == "NOT MEASURED"
    assert out["conditions"]["fine-tuned"]["vs base"] == {"only_this": 4, "only_other": 0, "p": 0.125}


def test_threeway_refuses_conditions_scored_on_different_rows():
    import threeway
    data = {"b": preds({"a": True, "b": False}), "f": preds({"a": True})}
    with pytest.raises(SystemExit):
        threeway.comparison({"base": "b", "fine-tuned": "f"}, loader=data.get)


@pytest.mark.parametrize("rag_correct, rag_abstain, expected, over", [
    ([1, 1, 1, 0], {}, "graceful", False),          # no lower than without RAG
    ([1, 0, 0, 0], {}, "harmful", False),           # the wrong table misled it
    ([1, 1, 1, 0], {"q3": True}, "graceful", True), # same accuracy, but it started refusing
])
def test_degradation_verdict(rag_correct, rag_abstain, expected, over):
    import threeway
    ids = ["q0", "q1", "q2", "q3"]
    no_rag = preds({i: i != "q3" for i in ids})
    rag = preds({i: bool(c) for i, c in zip(ids, rag_correct)}, rag_abstain)
    v = threeway.verdict(rag, no_rag, ids)
    assert (v["verdict"], v["over_abstention"]) == (expected, over)


def test_week3_report_draft_is_generated_and_says_what_is_missing():
    import report_week3
    text = report_week3.build()
    assert "1168 tables / 7675 rows" in text and "NOT MEASURED" in text
    assert (REPO / "docs/report_draft_week3.md").read_text(encoding="utf-8") == text


def test_gpu_runbook_commands_match_the_code():
    """Every config, split and eval-set name the Week 3 runbook submits must exist."""
    import re
    import evaluate
    text = (REPO / "docs/WEEK3_GPU_RUNBOOK.md").read_text(encoding="utf-8")
    for cfg in set(re.findall(r"configs/(\w+)\.yaml", text)):
        assert (REPO / f"configs/{cfg}.yaml").is_file(), cfg
    for split in set(re.findall(r"--split (\w+)", text)):
        assert split in ("val", "test", *evaluate.EVAL_SETS), split
    for sb in set(re.findall(r"slurm/(\w+)\.sbatch", text)):
        assert (REPO / f"slurm/{sb}.sbatch").is_file(), sb
