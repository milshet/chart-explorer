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
