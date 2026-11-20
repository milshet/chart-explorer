"""CPU-only checks for scripts/scoring.py and the parts of scripts/evaluate.py that do not need a GPU."""
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
import evaluate  # noqa: E402
import scoring  # noqa: E402


# --- scorer ----------------------------------------------------------------------------------------

def item(answer, answer_type, extra=None):
    gold = {"answer": answer, "answer_type": answer_type, "unit": None, **(extra or {})}
    return {"id": "x", "answer_type": answer_type, "input_text": "| a | 12.5 |",
            "messages": [{}, {}, {"role": "assistant", "content": json.dumps(gold)}]}


@pytest.mark.parametrize("raw, expected", [
    ('{"a": 1}', {"a": 1}),
    ('```json\n{"a": "}"}\n```', {"a": "}"}),                 # brace inside a string, code fence
    ('Answer: {"a": {"b": 2}} trailing {"c": 3}', {"a": {"b": 2}}),
    ('{"a": "x\\"}"}', {"a": 'x"}'}),                       # escaped quote
    ("no json here", None),
    ('{"a": 1', None),                                       # cut off (max_new_tokens)
])
def test_extract_json(raw, expected):
    assert scoring.extract_json(raw) == expected


@pytest.mark.parametrize("pred, gold, kind, ok", [
    ("5.6", 5.45, "number", True),        # within 5%
    ("6", 5.45, "number", False),
    ("$1,390", 1390, "number", True),
    (0, 0, "number", True),
    ("October.", "october", "text", True),
    ("Yes", "yes", "yes_no", True),
    (None, None, "unanswerable", True),
    ("null", None, "unanswerable", True),
    ("12", None, "unanswerable", False),
    (None, "october", "text", False),
])
def test_relaxed_match(pred, gold, kind, ok):
    assert scoring.relaxed_match(pred, gold, kind) is ok


def test_score_schema_depends_on_style():
    it = item(12.5, "number", {"evidence": ["Row=a, Col=b (12.5)"], "rationale": "read it"})
    short = '{"answer": 12.5, "answer_type": "number", "unit": null}'
    assert scoring.score(it, short, "answer_only")["schema_valid"]
    assert not scoring.score(it, short, "cot")["schema_valid"]     # cot also needs evidence + rationale
    full = scoring.score(it, it["messages"][2]["content"], "cot")
    assert full["correct"] and full["schema_valid"] and full["evidence_ok"] == full["evidence_n"] == 1
    with pytest.raises(ValueError):
        scoring.score(it, short, "reasoning")                      # a targets name, not a scorer style


@pytest.mark.parametrize("variant", ["image_only", "image_table"])
@pytest.mark.parametrize("targets", ["reasoning", "answer_only"])
@pytest.mark.parametrize("split", ["val", "test"])
def test_every_gold_answer_scores_full_marks(variant, targets, split):
    """The scorer must accept the dataset's own answers; otherwise accuracy has a ceiling below 100%."""
    sub = "answer_only/" if targets == "answer_only" else ""
    rows = [json.loads(l) for l in open(REPO / f"data/v2.0/{variant}/{sub}{split}.jsonl", encoding="utf-8")]
    style = evaluate.STYLE[targets]
    results = [scoring.score(r, r["messages"][2]["content"], style) for r in rows]
    overall, by = scoring.summarise(results, rows)
    assert overall["relaxed_accuracy"] == 1 and overall["schema_valid"] == 1
    assert sum(v["count"] for v in by["question_type"].values()) == len(rows)


def test_wrong_answers_lower_the_score():
    rows = [json.loads(l) for l in open(REPO / "data/v2.0/image_only/answer_only/val.jsonl", encoding="utf-8")]
    results = [scoring.score(r, '{"answer": "nonsense", "answer_type": "text", "unit": null}', "answer_only") for r in rows]
    overall, _ = scoring.summarise(results, rows)
    assert overall["relaxed_accuracy"] < 0.1 and overall["schema_valid"] == 1


def test_scorer_matches_the_notebook():
    """The notebook imports scoring.py instead of defining its own copy (one definition of "correct")."""
    nb = json.loads((REPO / "notebooks/01_vlm_baselines.ipynb").read_text(encoding="utf-8"))
    src = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
    assert any("import scoring" in s for s in src)
    assert not any("def relaxed_match" in s or "def extract_json" in s for s in src)
    assert any(f"FEW_SHOT_IDS = {json.dumps(evaluate.FEW_SHOT_IDS)}" in s for s in src)


# --- evaluate.py settings --------------------------------------------------------------------------

def fake_adapter(tmp_path, as_epoch=False, **cfg):
    run = tmp_path / "run_x"
    d = run / ("checkpoints/epoch-1" if as_epoch else "adapter")
    d.mkdir(parents=True)
    (d / "adapter_model.safetensors").write_bytes(b"w")
    conf = {"run_name": "run_x", "data_dir": "data/v2.0", "variant": "image_only", "targets": "reasoning",
            "max_pixels": 800 * 800, **cfg}
    (run / "config.json" if as_epoch else d / "train_config.json").write_text(json.dumps(conf))
    return d


@pytest.mark.parametrize("as_epoch", [False, True])
def test_adapter_settings_come_from_the_run(tmp_path, as_epoch):
    d = fake_adapter(tmp_path, as_epoch)
    s = evaluate.resolve(evaluate.parse_args(["--adapter", str(d), "--split", "val"]))
    assert (s["targets"], s["style"], s["variant"]) == ("reasoning", "cot", "image_only")
    assert s["name"] == f"run_x-{d.name}_val_image_only_cot_0shot_bf16"


@pytest.mark.parametrize("argv", [
    ["--split", "val"],                                             # base model needs --targets
    ["--split", "val", "--targets", "answer_only", "ADAPTER"],      # conflicts with the run's targets
    ["--split", "test", "ADAPTER"],                                 # test needs --final-test
])
def test_resolve_rejects_mistakes(tmp_path, argv):
    d = fake_adapter(tmp_path)
    argv = [a for a in argv if a != "ADAPTER"] + (["--adapter", str(d)] if "ADAPTER" in argv else [])
    with pytest.raises(SystemExit):
        evaluate.resolve(evaluate.parse_args(argv))


def test_adapter_needs_weights_and_config(tmp_path):
    (tmp_path / "empty").mkdir()
    with pytest.raises(SystemExit):
        evaluate.adapter_config(tmp_path / "empty")
    d = fake_adapter(tmp_path)
    (d / "train_config.json").unlink()
    with pytest.raises(SystemExit):
        evaluate.adapter_config(d)


def test_pinned_shots_exist_in_both_train_files():
    for sub in ("", "answer_only/"):
        ids = {json.loads(l)["id"] for l in open(REPO / f"data/v2.0/image_only/{sub}train.jsonl", encoding="utf-8")}
        assert set(evaluate.FEW_SHOT_IDS) <= ids


def test_answer_length_and_truncation():
    assert evaluate.answer_length([5, 6, 2, 0, 0], {2, 0}) == (2, False)
    assert evaluate.answer_length([5, 6, 7], {2}) == (3, True)


# --- compare.py ------------------------------------------------------------------------------------

def test_mcnemar_exact():
    import compare
    a = {i: i < 8 for i in range(20)}             # right on 0-7
    b = {i: 4 <= i < 16 for i in range(20)}       # right on 4-15
    x, y, p = compare.mcnemar(a, b)
    assert (x, y) == (4, 8) and p == pytest.approx(0.3877, abs=1e-4)
    assert compare.mcnemar(a, a)[2] == 1.0
    with pytest.raises(ValueError):
        compare.mcnemar(a, {0: True})


def test_compare_table_from_committed_results():
    import compare
    evals = compare.load(REPO / "results" / "evals", "val")
    out = compare.table(evals, "run_a_r16-adapter_val_image_only_cot_0shot_bf16")
    assert "81.5%" in out and "reference" in out and "4 / 8" not in out
    assert "8 / 4 (p = 0.388)" in out              # Run B vs Run A, as in results/evals/README.md


# --- abstention pair (task 3.0b) -------------------------------------------------------------------

def test_abstention_is_a_pair_and_counts_refusals_on_answerable_items():
    groups = {"question_type": "q", "source": "s", "chart_type": "c"}
    una, ans = {**item(None, "unanswerable"), **groups}, {**item(12.5, "number"), **groups, "id": "y"}
    refuse = '{"answer": null, "answer_type": "unanswerable", "unit": null}'
    guess = '{"answer": 3, "answer_type": "number", "unit": null}'
    # refusing an answerable item: wrong, like a guess, but now visible as a false abstention
    results = [scoring.score(una, refuse, "answer_only"), scoring.score(ans, refuse, "answer_only")]
    overall, _ = scoring.summarise(results, [una, ans])
    assert overall["abstention"] == {"true_abstention": 1.0, "n_unanswerable": 1,
                                     "false_abstention": 1.0, "n_answerable": 1}
    assert not results[1]["correct"]
    guessed, _ = scoring.summarise([scoring.score(una, refuse, "answer_only"), scoring.score(ans, guess, "answer_only")],
                                   [una, ans])
    assert guessed["abstention"]["false_abstention"] == 0.0 and guessed["answerable_accuracy"] == overall["answerable_accuracy"]


def test_neither_abstention_rate_is_reported_alone():
    rows = [json.loads(l) for l in open(REPO / "data/v2.0/image_only/answer_only/val.jsonl", encoding="utf-8")]
    overall, _ = scoring.summarise([scoring.score(r, r["messages"][2]["content"], "answer_only") for r in rows], rows)
    assert "refusal_accuracy" not in overall and "true_abstention" not in overall and "false_abstention" not in overall
    assert set(overall["abstention"]) == {"true_abstention", "n_unanswerable", "false_abstention", "n_answerable"}
    assert overall["abstention"]["true_abstention"] == 1 and overall["abstention"]["false_abstention"] == 0


def test_unparseable_output_is_not_a_refusal():
    res = scoring.score(item(None, "unanswerable"), "I cannot tell.", "answer_only")
    assert not res["abstained"] and not res["correct"]


def test_committed_evaluations_carry_the_pair_and_rescore_without_moving():
    import rescore
    for folder in sorted((REPO / "results" / "evals").glob("*/summary.json")):
        summary, _, problems = rescore.rescore(folder.parent)
        assert problems == [], folder.parent.name
        committed = json.loads(folder.read_text(encoding="utf-8"))["overall"]
        assert "refusal_accuracy" not in committed and rescore.same(committed["abstention"], summary["overall"]["abstention"])


def test_rescore_refuses_a_moved_number(tmp_path):
    import shutil
    import rescore
    src = REPO / "results/evals/run_a_r16-adapter_val_image_only_cot_0shot_bf16"
    dst = tmp_path / src.name
    shutil.copytree(src, dst)
    rows = [json.loads(l) for l in open(dst / "predictions.jsonl", encoding="utf-8")]
    rows[0]["correct"] = not rows[0]["correct"]          # a verdict the current scorer would not give
    (dst / "predictions.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    assert rescore.rescore(dst)[2]


# --- Eval-C (task 3.0c) ----------------------------------------------------------------------------

def eval_c_rows(rel="image_only/eval_c.jsonl"):
    return [json.loads(l) for l in open(REPO / "data/eval_c" / rel, encoding="utf-8")]


def test_eval_c_prompts_are_built_exactly_like_the_frozen_dataset():
    import build_eval_c as b
    parts = b.prompt_parts()
    for (variant, targets), rel in b.FILES.items():
        for r in map(json.loads, open(REPO / "data/v2.0" / rel.replace("eval_c", "test"), encoding="utf-8")):
            built = b.messages(parts, variant, targets, r["image"], r["question"], r["input_text"])
            assert built[:2] == r["messages"][:2], r["id"]
            if r["answer_type"] == "unanswerable":
                assert json.loads(built[2]["content"]) == json.loads(r["messages"][2]["content"])


def test_committed_eval_c_is_what_the_builder_produces():
    import build_eval_c as b
    probes, corpus = b.build()
    recs = b.records(probes, corpus)
    for key, rel in b.FILES.items():
        assert recs[key] == eval_c_rows(rel)
    manifest = dict(reversed(l.split()) for l in (REPO / "data/eval_c/MANIFEST.txt").read_text().splitlines() if l)
    import hashlib
    for rel, digest in manifest.items():
        assert hashlib.sha256((REPO / "data/eval_c" / rel).read_bytes()).hexdigest() == digest


def test_eval_c_reasons_are_proved_by_the_tables():
    import build_eval_c as b
    from rag import tables
    corpus = {c["chart_id"]: c for c in tables.load_corpus()[0]}
    rows = eval_c_rows()
    assert len(rows) == 50 and {r["abstain_reason"] for r in rows} == set(b.QUOTA)
    for r in rows:
        assert all(not b.occurs(e, r["input_text"]) for e in r["absent_entities"]), r["id"]
        assert all(b.occurs(e, r["input_text"]) for e in r["present_entities"]), r["id"]
        assert corpus[r["chart_id"]]["image_split"] in ("val", "test")       # never a training chart
        if r["origin"] == "catalogue":                                       # the question really names them
            source = corpus[r["question_source_chart"]]["text"]
            assert all(b.occurs(e, r["question"]) and b.occurs(e, source) for e in r["absent_entities"]), r["id"]


def test_eval_c_questions_were_never_selected_into_the_dataset():
    used = {json.loads(l)["question"] for s in ("train", "val", "test")
            for l in open(REPO / f"data/v2.0/selection/{s}.jsonl", encoding="utf-8")}
    assert not any(r["question"] in used for r in eval_c_rows() if r["origin"] == "catalogue")


@pytest.mark.parametrize("variant", ["image_only", "image_table"])
@pytest.mark.parametrize("targets", ["reasoning", "answer_only"])
def test_eval_c_loads_through_evaluate_and_gold_scores_full_marks(variant, targets):
    s = evaluate.resolve(evaluate.parse_args(["--split", "eval_c", "--targets", targets, "--variant", variant]))
    assert s["items_dir"] == "data/eval_c" and s["data_dir"] == "data/v2.0"
    items, shots, fp = evaluate.load_rows(s, "eval_c", evaluate.parse_args(["--split", "eval_c", "--targets", targets]))
    assert len(items) == 50 and fp["data_dir"] == "data/eval_c" and fp["images"] == len({r["image"] for r in items})
    res = [scoring.score(r, r["messages"][2]["content"], s["style"]) for r in items]
    overall, _ = scoring.summarise(res, items)
    assert overall["abstention"]["true_abstention"] == 1 and overall["abstention"]["n_answerable"] == 0


def test_eval_c_few_shot_baseline_fingerprints_both_sets():
    argv = ["--split", "eval_c", "--targets", "answer_only", "--shots", "2"]
    s = evaluate.resolve(evaluate.parse_args(argv))
    _, shots, fp = evaluate.load_rows(s, "eval_c", evaluate.parse_args(argv))
    assert [r["id"] for r in shots] == evaluate.FEW_SHOT_IDS and fp["shots"]["data_dir"] == "data/v2.0"


def test_adapter_on_eval_c_needs_final_test(tmp_path):
    d = fake_adapter(tmp_path)
    with pytest.raises(SystemExit):
        evaluate.resolve(evaluate.parse_args(["--split", "eval_c", "--adapter", str(d)]))
    s = evaluate.resolve(evaluate.parse_args(["--split", "eval_c", "--adapter", str(d), "--final-test"]))
    assert s["items_dir"] == "data/eval_c" and s["name"].endswith("_eval_c_image_only_cot_0shot_bf16")


# --- error taxonomy (task 3.0e) --------------------------------------------------------------------

def tax_item(answer, answer_type, qtype="retrieval", table="| a | 40 | 60 |"):
    gold = {"evidence": [], "rationale": "r", "answer": answer, "answer_type": answer_type, "unit": None}
    return {"id": "x", "answer_type": answer_type, "question_type": qtype, "input_text": table,
            "messages": [{}, {}, {"role": "assistant", "content": json.dumps(gold)}]}


def out(answer, answer_type="number", evidence=("40", "60")):
    return json.dumps({"evidence": list(evidence), "rationale": "r", "answer_type": answer_type, "unit": None,
                       "answer": answer})


@pytest.mark.parametrize("item_args, raw, category", [
    ((100, "number"), out(100), "ok"),
    ((100, "number"), "no json", "schema_invalid"),
    ((None, "unanswerable", "unanswerable"), out(5), "failed_to_abstain"),
    ((100, "number"), out(None, "unanswerable"), "false_abstention"),
    (("yes", "yes_no"), out(100), "answer_type_mismatch"),
    ((64, "number"), out(0.64), "scale_error"),
    ((100, "number"), out(110), "near_miss"),
    ((100, "number", "arithmetic_comparison"), out(20), "reasoning_error"),        # read 40 and 60 right
    ((100, "number", "arithmetic_comparison"), out(20, evidence=("7",)), "extraction_error"),
    (("france", "text"), out("Spain", "text"), "text_mismatch"),
])
def test_taxonomy_categories(item_args, raw, category):
    import taxonomy
    assert taxonomy.classify({"raw": raw}, tax_item(*item_args), "cot")[0] == category


def test_schema_invalid_says_which_rule_broke():
    import taxonomy
    raw = '{"answer": 1, "answer_type": "count", "evidence": "40"}'
    assert taxonomy.classify({"raw": raw}, tax_item(1, "number"), "cot")[1] == \
        ["missing:rationale", "missing:unit", "bad_answer_type", "evidence_not_list"]


def test_flags_are_not_categories():
    import taxonomy
    cat, _, flags = taxonomy.classify({"raw": out(100, evidence=("999",))}, tax_item(100, "number"), "cot")
    assert cat == "ok" and flags == ["ungrounded_evidence"]
    probe = {**tax_item(None, "unanswerable", "unanswerable"), "absent_entities": ["Germany"]}
    assert taxonomy.classify({"raw": out(None, "unanswerable", ())}, probe, "cot")[2] == ["refusal_reason_unstated"]


def test_taxonomy_reconciles_with_every_committed_evaluation():
    import taxonomy
    for folder in sorted((REPO / "results/evals").glob("*/summary.json")):
        t = taxonomy.taxonomy(folder.parent)            # exits if share(ok) != relaxed accuracy
        assert sum(t["categories"].values()) == t["n"]
        committed = folder.parent / "taxonomy.json"
        assert committed.is_file() and json.loads(committed.read_text(encoding="utf-8")) == t


# --- serving smoke (task 3.0f) ---------------------------------------------------------------------

def test_serve_smoke_probe_is_answered_correctly_by_run_a_on_val():
    import serve_smoke
    item = serve_smoke.probe()
    preds = {json.loads(l)["id"]: json.loads(l) for l in
             open(REPO / "results/evals/run_a_r16-adapter_val_image_only_cot_0shot_bf16/predictions.jsonl", encoding="utf-8")}
    assert scoring.score(item, preds[item["id"]]["raw"], "cot")["correct"]


def test_serve_smoke_builds_an_openai_multimodal_request():
    import serve_smoke
    item = serve_smoke.probe()
    msgs = serve_smoke.openai_messages(item, REPO / "data/v2.0" / item["image"])
    assert msgs[0] == {"role": "system", "content": item["messages"][0]["content"]}
    kinds = [c["type"] for c in msgs[1]["content"]]
    assert kinds == ["image_url", "text"] and msgs[1]["content"][0]["image_url"]["url"].startswith("data:image/png;base64,")
    assert msgs[1]["content"][1] == item["messages"][1]["content"][1]


# --- LLM-as-judge harness (task 3.7); no API calls ------------------------------------------------

@pytest.mark.parametrize("text, ok", [
    ('{"correctness": "yes", "grounding": "n/a", "abstention": "yes", "note": "x"}', True),
    ('Sure! {"correctness": "Yes", "grounding": "no", "abstention": "no"}', True),
    ('{"correctness": "maybe", "grounding": "no", "abstention": "yes"}', False),   # not in the vocabulary
    ('{"correctness": "yes", "grounding": "no", "abstention": "n/a"}', False),     # abstention is always graded
    ("I think it is correct.", False),
    (None, False),                                                                 # judge refused
])
def test_judge_parse_excludes_unreadable_verdicts(text, ok):
    import judge
    assert (judge.parse(text) is not None) is ok


def test_judge_summary_excludes_rather_than_scores_zero():
    import judge
    v = {"correctness": "yes", "grounding": "yes", "abstention": "yes"}
    judged = [{"id": "a", "verdict": v, "scorer_evidence": [2, 2]}, {"id": "b", "verdict": None, "scorer_evidence": [0, 0]},
              {"id": "c", "verdict": {**v, "grounding": "yes"}, "scorer_evidence": [0, 3]}]
    s = judge.summarise(judged)
    assert s["n_excluded"] == 1 and s["correctness"] == {"yes": 2, "n": 2}
    assert s["grounding_disagreements"] == ["c"]           # judge says grounded, scorer found none of 3 in the table


def test_kappa_exposes_always_yes_on_a_skewed_axis():
    import judge
    truth = ["yes"] * 9 + ["no"]
    n, raw, kappa = judge.cohen(["yes"] * 10, truth)
    assert raw == 0.9 and kappa == 0.0


def test_judge_sample_mixes_eval_a_and_eval_c():
    import judge
    rows, missing = judge.draw("run_a_r16-adapter")
    assert missing == []
    test = [r for r in rows if r["split"] == "test"]
    evc = [r for r in rows if r["split"] == "eval_c"]
    assert len(test) == judge.SAMPLE["test"] and sum(r["gold"] is None for r in test) == 9  # all nine unanswerable
    assert len(evc) == judge.SAMPLE["eval_c"] and all(r["gold"] is None for r in evc)


def test_judge_grades_the_served_condition_of_exactly_that_run():
    import judge
    # run_a_r16 must not pick up run_a_r16_table / _vision or its own RAG folder, all of which sort near it
    assert judge.eval_folder("run_a_r16", "test").name == "run_a_r16-adapter_test_image_only_cot_0shot_bf16"
    assert judge.eval_folder("run_a_r16", "eval_c").name == "run_a_r16-adapter_eval_c_image_only_cot_0shot_bf16"
    with pytest.raises(ValueError):
        judge.eval_folder("base", "test")                  # three prompt settings: ambiguous, so it refuses


def test_judge_sample_reports_a_missing_split(monkeypatch):
    import judge
    real = judge.eval_folder
    monkeypatch.setattr(judge, "eval_folder", lambda name, split: None if split == "eval_c" else real(name, split))
    rows, missing = judge.draw("run_a_r16-adapter")
    assert missing == ["eval_c"]                            # not scored on Eval-C: said, not silently skipped
    assert {r["split"] for r in rows} == {"test"}


# --- local judge backend (task 3.13); no model is loaded -------------------------------------------

@pytest.mark.parametrize("raw, final", [
    ('<|channel|>analysis<|message|>maybe {"correctness": "no"}<|end|><|start|>assistant<|channel|>final<|message|>'
     '{"correctness": "yes", "grounding": "yes", "abstention": "yes"}<|return|>',
     '{"correctness": "yes", "grounding": "yes", "abstention": "yes"}'),        # reasoning's JSON is not the verdict
    ('<|channel|>analysis<|message|>still thinking {"correctness": "no"', ""),   # cut off before the final channel
    ('{"correctness": "no", "grounding": "n/a", "abstention": "yes"}<|endoftext|>',
     '{"correctness": "no", "grounding": "n/a", "abstention": "yes"}'),         # a model without channels
])
def test_judge_reads_only_the_final_channel(raw, final):
    import judge
    assert judge.final_text(raw) == final


def test_local_judge_rows_exclude_cut_off_replies_and_keep_raw():
    import judge
    v = '{"correctness": "yes", "grounding": "no", "abstention": "yes"}'
    replies = {"a": f"<|channel|>final<|message|>{v}<|return|>", "b": "<|channel|>analysis<|message|>hmm", "c": None}
    rows = [{"id": k, "scorer_evidence": [1, 1]} for k in replies]
    judged = judge.judge_rows(rows, lambda r: (replies[r["id"]], {"new_tokens": 7}))
    assert [bool(j["verdict"]) for j in judged] == [True, False, False]
    assert judged[1]["judge_raw"] == replies["b"] and judged[0]["new_tokens"] == 7
    assert judge.summarise(judged)["n_excluded"] == 2


def test_judge_defaults_to_the_free_local_model():
    import judge
    assert judge.LOCAL_MODEL == "arena-education/Llama-3.1-8B-Instruct-bnb-8bit" and "qwen" not in judge.LOCAL_MODEL.lower()
    s = json.loads((REPO / "results/judge/run_a_r16/summary.json").read_text(encoding="utf-8"))
    assert s["judge_model"] == judge.LOCAL_MODEL                         # a rerun with defaults reproduces it
    text = (REPO / "slurm/judge.sbatch").read_text(encoding="utf-8")
    assert "HF_HUB_OFFLINE=1" in text and 'scripts/judge.py run "$@"' in text


# --- served latency benchmark (tasks 3.8, 3.14); no model is loaded --------------------------------

def test_serve_bench_times_every_adapter_on_the_same_val_requests():
    import serve_bench
    import smoke_test as st
    cot = st.read_jsonl(REPO / "data/v2.0/image_only/val.jsonl", None)
    ans = st.read_jsonl(REPO / "data/v2.0/image_only/answer_only/val.jsonl", None)
    (w1, t1), (w2, t2) = serve_bench.bench_items(cot), serve_bench.bench_items(list(reversed(ans)))
    assert [i["id"] for i in w1 + t1] == [i["id"] for i in w2 + t2]     # same requests, same order, any file order
    assert (len(w1), len(t1)) == (serve_bench.WARMUP, serve_bench.N_REQUESTS)
    assert not {i["id"] for i in w1} & {i["id"] for i in t1}           # warm-up requests are not timed


def test_serve_bench_summary_uses_observed_percentiles():
    import serve_bench
    assert serve_bench.pct(list(range(1, 101)), 50) == 50 and serve_bench.pct(list(range(1, 101)), 95) == 95
    assert serve_bench.pct([3.0], 95) == 3.0
    rows = [{"e2e_s": e, "ttft_s": 0.1, "decode_tok_s": 60.0, "new_tokens": 10, "generate_s": 0.5, "prompt_tokens": 900,
             "correct": c, "schema_valid": True} for e, c in ((1.0, True), (2.0, False), (4.0, True))]
    s = serve_bench.summarise(rows)
    assert s["e2e_s"]["p50"] == 2.0 and s["e2e_s"]["p95"] == 4.0 and s["overall_tok_s"] == 20.0
    assert s["accuracy_on_these_items"] == round(2 / 3, 4)
