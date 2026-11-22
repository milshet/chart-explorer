"""Task 4.1 on the laptop: the base check, the weight check and the answer comparison, without a GPU.

The weight check is exercised on small bf16 tensors with realistic magnitudes (deltas comparable to one unit in
the last place of the weights), so the failures it exists for (wrong scaling, a module the merge skipped, a dtype
change, a weight outside the adapter touched) are each shown to raise, and a correct PEFT-style merge to pass.
"""
import json
import struct
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
from serve import generation as gen  # noqa: E402
from serve import merge as M  # noqa: E402

BASE = "Qwen/Qwen2.5-VL-3B-Instruct"
REV = "66285546d2b821cf421d4f5eb2576359d3770cd3"


def adapter(cfg=None, trained_revision=REV, train_config=None, targets=None):
    c = {"peft_type": "LORA", "base_model_name_or_path": BASE, "revision": None, "r": 16, "lora_alpha": 32,
         "use_rslora": False, "use_dora": False, "modules_to_save": None, "rank_pattern": {}, "alpha_pattern": {},
         "fan_in_fan_out": False, **(cfg or {})}
    return {"dir": Path("/x/adapter"), "config": c, "weights": Path("/x/adapter/adapter_model.safetensors"),
            "train_config": train_config if train_config is not None else
            {"lora": {"r": 16, "alpha": 32}, "max_pixels": 640000},
            "trained_revision": trained_revision, "targets": targets}


# --- 1. base check --------------------------------------------------------------------------------------------

def test_check_base_passes_and_reports_scaling():
    got = M.check_base(adapter(), REV, expected=BASE)
    assert got["revision_verified"] and got["scaling"] == 2.0 and got["base_revision"] == REV


@pytest.mark.parametrize("cfg, trained, match", [
    ({"base_model_name_or_path": "Qwen/Qwen2.5-VL-7B-Instruct"}, REV, "trained against"),
    ({"revision": "0" * 40}, REV, "adapter_config.revision"),
    ({}, "1" * 40, "run_summary.model_revision"),
    ({"base_model_name_or_path": f"/hf/models--Qwen--Qwen2.5-VL-3B-Instruct/snapshots/{'2' * 40}"}, REV,
     "base_model_name_or_path"),
    ({"use_dora": True}, REV, "use_dora"),
    ({"modules_to_save": ["lm_head"]}, REV, "modules_to_save"),
    ({"rank_pattern": {"q_proj": 8}}, REV, "rank_pattern"),
    ({"peft_type": "IA3"}, REV, "only LoRA"),
    ({"fan_in_fan_out": True}, REV, "fan_in_fan_out"),
])
def test_check_base_raises_on_a_different_base_or_an_unverifiable_adapter(cfg, trained, match):
    with pytest.raises(M.MergeError, match=match):
        M.check_base(adapter(cfg, trained), REV, expected=BASE)


def test_unrecorded_revision_raises_unless_allowed_and_then_says_so():
    with pytest.raises(M.MergeError, match="nothing records"):
        M.check_base(adapter(trained_revision=None), REV, expected=BASE)
    got = M.check_base(adapter(trained_revision=None), REV, expected=BASE, allow_unrecorded_revision=True)
    assert got["revision_verified"] is False


def test_snapshot_path_with_the_right_revision_counts_as_verified():
    a = adapter({"base_model_name_or_path": f"/hf/models--Qwen--Qwen2.5-VL-3B-Instruct/snapshots/{REV}"},
                trained_revision=None)
    assert M.check_base(a, REV, expected=BASE)["revision_verified"]


def test_adapter_from_another_run_raises():
    with pytest.raises(M.MergeError, match="not this run's adapter"):
        M.check_base(adapter(train_config={"lora": {"r": 64, "alpha": 128}}), REV, expected=BASE)
    with pytest.raises(M.MergeError, match="max_pixels"):
        M.check_base(adapter(train_config={"max_pixels": 1000}), REV, expected=BASE)


def test_lora_scaling():
    assert M.lora_scaling({"r": 16, "lora_alpha": 32}) == 2.0
    assert M.lora_scaling({"r": 16, "lora_alpha": 32, "use_rslora": True}) == 8.0


def test_read_adapter(tmp_path):
    (tmp_path / "adapter").mkdir()
    with pytest.raises(M.MergeError, match="adapter_config"):
        M.read_adapter(tmp_path / "adapter")
    (tmp_path / "adapter/adapter_config.json").write_text(json.dumps({"r": 16}))
    with pytest.raises(M.MergeError, match="safetensors"):
        M.read_adapter(tmp_path / "adapter")
    (tmp_path / "adapter/adapter_model.safetensors").write_bytes(b"")
    (tmp_path / "adapter/targets.json").write_text(json.dumps({"modules": ["base_model.model.a.q_proj"]}))
    (tmp_path / "run_summary.json").write_text(json.dumps({"model_revision": REV}))
    a = M.read_adapter(tmp_path / "adapter")
    assert a["trained_revision"] == REV and a["targets"] == ["base_model.model.a.q_proj"]
    assert a["train_config"] is None


# --- 2. adapter file and targets ----------------------------------------------------------------------------------

def write_safetensors(path, tensors):
    """A real .safetensors file without torch: {name: (dtype, shape)} filled with zeros."""
    header, offset = {"__metadata__": {"format": "pt"}}, 0
    for name, (dtype, shape) in tensors.items():
        size = 4
        for d in shape:
            size *= d
        header[name] = {"dtype": dtype, "shape": list(shape), "data_offsets": [offset, offset + size]}
        offset += size
    blob = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(blob)) + blob + b"\0" * offset)


def test_safetensors_header_reads_names_without_torch(tmp_path):
    write_safetensors(tmp_path / "a.safetensors", {"x.lora_A.weight": ("F32", (2, 3))})
    h = M.safetensors_header(tmp_path / "a.safetensors")
    assert list(h) == ["x.lora_A.weight"] and h["x.lora_A.weight"]["shape"] == [2, 3]


def test_factor_modules_strips_the_peft_prefix_and_pairs_a_with_b():
    keys = [f"base_model.model.model.language_model.layers.0.self_attn.q_proj.lora_{x}.weight" for x in "AB"]
    keys += ["base_model.model.model.language_model.layers.0.mlp.up_proj.lora_A.default.weight",
             "base_model.model.model.language_model.layers.0.mlp.up_proj.lora_B.default.weight"]
    mods = M.factor_modules(keys)
    assert set(mods) == {"model.language_model.layers.0.self_attn.q_proj", "model.language_model.layers.0.mlp.up_proj"}
    assert mods["model.language_model.layers.0.self_attn.q_proj"]["A"] == keys[0]


def test_factor_modules_raises_on_non_lora_tensors_and_lone_factors():
    with pytest.raises(M.MergeError, match="non-LoRA"):
        M.factor_modules(["base_model.model.lm_head.weight"])
    with pytest.raises(M.MergeError, match="one LoRA factor"):
        M.factor_modules(["base_model.model.a.q_proj.lora_A.weight"])


def test_check_targets_matches_the_file_against_what_training_wrapped():
    mods = {"model.language_model.layers.0.self_attn.q_proj": {}, "model.visual.blocks.0.attn.qkv": {}}
    got = M.check_targets(mods, [f"base_model.model.{m}" for m in mods])
    assert got["modules"] == 2 and got["by_part"] == {"decoder": 1, "vision_tower": 1}
    with pytest.raises(M.MergeError, match="disagree"):
        M.check_targets(mods, ["base_model.model.model.language_model.layers.0.self_attn.q_proj"])
    assert M.check_targets(mods, None)["targets_json"] is False


# --- 2. weight check --------------------------------------------------------------------------------------------

torch = pytest.importorskip("torch")


def toy(seed=0, scaling=2.0):
    """A base with two LoRA modules and two untouched tensors, its factors, and a PEFT-style merge of it."""
    g = torch.Generator().manual_seed(seed)
    bf = torch.bfloat16
    before = {"m.q.weight": (torch.randn(32, 64, generator=g) * 0.02).to(bf),
              "m.v.weight": (torch.randn(48, 64, generator=g) * 0.02).to(bf),
              "m.norm.weight": torch.ones(64, dtype=bf),
              "m.visual.w": (torch.randn(16, 16, generator=g) * 0.02).to(bf)}
    factors = {"m.q": (torch.randn(4, 64, generator=g) * 0.01, torch.randn(32, 4, generator=g) * 0.01),
               "m.v": (torch.randn(4, 64, generator=g) * 0.01, torch.randn(48, 4, generator=g) * 0.01)}
    return before, factors, peft_merge(before, factors, scaling)


def peft_merge(before, factors, scaling):
    """What PEFT does on CPU: delta in fp32, rounded to bf16, then added to the bf16 weight."""
    after = {k: v.clone() for k, v in before.items()}
    for m, (a, b) in factors.items():
        after[f"{m}.weight"] = after[f"{m}.weight"] + (scaling * (b @ a)).to(torch.bfloat16)
    return after


def test_a_correct_merge_passes_and_most_elements_change():
    before, factors, after = toy()
    got = M.check_merged_weights(before, after, factors, 2.0)
    assert got["modules_checked"] == 2 and got["tensors_unchanged"] == 2 and got["dtype"] == "bfloat16"
    assert got["elements_changed_by_merge"] > 0.5 * got["elements_in_modules"]
    assert got["elements_rounded_differently"] <= M.WEIGHT_MISMATCH_MAX * got["elements_changed_by_merge"]
    assert 0.9 < got["coefficient_min"] <= got["coefficient_max"] < 1.1


def test_wrong_scaling_raises():
    before, factors, _ = toy()
    with pytest.raises(M.MergeError, match="differ from base"):
        M.check_merged_weights(before, peft_merge(before, factors, 1.0), factors, 2.0)  # alpha/r read as 1


def test_a_module_the_merge_skipped_raises():
    before, factors, after = toy()
    after["m.v.weight"] = before["m.v.weight"].clone()
    with pytest.raises(M.MergeError, match="m.v.weight"):
        M.check_merged_weights(before, after, factors, 2.0)


def test_unmerged_weights_raise():
    before, factors, _ = toy()
    with pytest.raises(M.MergeError):
        M.check_merged_weights(before, {k: v.clone() for k, v in before.items()}, factors, 2.0)


def test_a_change_outside_the_adapter_raises():
    before, factors, after = toy()
    after["m.visual.w"] = after["m.visual.w"] * 1.01
    with pytest.raises(M.MergeError, match="outside the LoRA modules"):
        M.check_merged_weights(before, after, factors, 2.0)


def test_a_dtype_change_raises():
    before, factors, after = toy()
    after["m.norm.weight"] = after["m.norm.weight"].float()
    with pytest.raises(M.MergeError, match="changed dtype"):
        M.check_merged_weights(before, after, factors, 2.0)


def test_leftover_lora_keys_and_missing_keys_raise():
    before, factors, after = toy()
    after["m.q.lora_A.default.weight"] = factors["m.q"][0]
    del after["m.norm.weight"]
    with pytest.raises(M.MergeError, match="1 missing.*1 extra"):
        M.check_merged_weights(before, after, factors, 2.0)


def test_adapter_naming_a_module_the_base_lacks_raises():
    before, factors, after = toy()
    factors["m.k"] = factors["m.q"]
    with pytest.raises(M.MergeError, match="name no weight in the base"):
        M.check_merged_weights(before, after, factors, 2.0)


def test_shape_mismatch_raises():
    before, factors, after = toy()
    factors["m.q"] = (factors["m.q"][0][:, :32], factors["m.q"][1])
    with pytest.raises(M.MergeError, match="cannot take"):
        M.check_merged_weights(before, after, factors, 2.0)


# --- 3. answers ----------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("raw, parsed", [
    ('{"answer": "1,390", "answer_type": "number"}', ("num", 1390.0)),
    ('{"answer": 1390}', ("num", 1390.0)),
    ('{"answer": "$1390."}', ("num", 1390.0)),
    ('{"answer": null}', ("null",)),
    ('{"answer": "None"}', ("null",)),
    ('{"answer": " October. "}', ("text", "october")),
    ('{"answer": true}', ("text", "yes")),
    ("not json", ("invalid",)),
    ('["a list"]', ("invalid",)),
])
def test_parsed_answer(raw, parsed):
    assert gen.parsed_answer(raw) == parsed


@pytest.mark.parametrize("a, b, same", [
    ('{"answer": "1,390"}', '{"evidence": ["x"], "answer": 1390.0}', True),   # rationale differs, answer does not
    ('{"answer": 5.45}', '{"answer": 5.46}', False),                         # within the scorer's 5%: still two answers
    ('{"answer": "Yes"}', '{"answer": "yes"}', True),
    ('{"answer": null}', '{"answer": "null"}', True),
    ('{"answer": null}', '{"answer": 0}', False),
    ("garbage", "other garbage", True),                                     # both failed the format the same way
    ("garbage", '{"answer": null}', False),
])
def test_same_answer_compares_parsed_answers_not_strings(a, b, same):
    assert gen.same_answer(gen.parsed_answer(a), list(gen.parsed_answer(b))) is same


def row(i, answer, correct=True, schema=True, raw=None):
    raw = raw if raw is not None else json.dumps({"answer": answer})
    return {"id": f"i{i}", "raw": raw, "answer": list(gen.parsed_answer(raw)), "correct": correct,
            "schema_valid": schema}


def test_compare_answers_passes_within_tolerance():
    ref = [row(i, i) for i in range(10)]
    merged = [row(i, i) for i in range(7)] + [row(i, i + 0.5, correct=False) for i in range(7, 10)]
    got = M.compare_answers(ref, merged, max_disagree=3)
    assert got["passed"] and got["disagree"] == 3 and got["broken_by_merge"] == 3
    assert got["correct"] == {"reference": 10, "merged": 7} and got["raw_identical"] == 7


def test_compare_answers_fails_past_tolerance_or_on_schema_loss():
    ref = [row(i, i) for i in range(10)]
    assert not M.compare_answers(ref, [row(i, -1) for i in range(10)], max_disagree=3)["passed"]
    worse = [row(i, i, schema=i > 1) for i in range(10)]
    assert not M.compare_answers(ref, worse, max_disagree=3, max_schema_drop=1)["passed"]


def test_compare_answers_needs_identical_items():
    with pytest.raises(ValueError):
        M.compare_answers([row(0, 1)], [row(1, 1)])


def test_sample_is_deterministic_and_independent_of_item_order():
    items = [{"id": f"val-{i}"} for i in range(300)]
    a = M.sample_items(items)
    assert len(a) == M.SAMPLE_N and len({i["id"] for i in a}) == M.SAMPLE_N
    assert [i["id"] for i in M.sample_items(list(reversed(items)))] == [i["id"] for i in a]
    assert len(M.sample_items(items[:5])) == 5


def test_tolerances_are_fixed_constants():
    assert (M.SAMPLE_N, M.MAX_DISAGREE, M.MAX_SCHEMA_DROP) == (100, 3, 1)


class ScriptedEngine:
    """Stands in for HFEngine: answers from a script, records what it was asked."""

    def __init__(self, script):
        self.script, self.calls = script, []

    def generate(self, messages, images, max_new_tokens=gen.MAX_NEW_TOKENS):
        self.calls.append((messages, max_new_tokens))
        return gen.Generation(self.script[messages], prompt_tokens=10, new_tokens=5, truncated=False, seconds=0.1)


def scored_item(i, gold):
    return {"id": f"i{i}", "answer_type": "number", "input_text": "| a | 12 |",
            "messages": [{}, {}, {"role": "assistant", "content": json.dumps(
                {"answer": gold, "answer_type": "number", "unit": None, "evidence": [], "rationale": ""})}]}


def test_answer_items_scores_each_answer_with_the_reporting_scorer():
    items = [scored_item(0, 12), scored_item(1, 7)]
    out = '{"evidence": ["a 12"], "rationale": "r", "answer": 12, "answer_type": "number", "unit": null}'
    engine = ScriptedEngine({"i0": out, "i1": out})
    rows = gen.answer_items(engine, items, {"style": "cot"}, log=None, build=lambda it, s: (it["id"], []))
    assert [r["correct"] for r in rows] == [True, False] and all(r["schema_valid"] for r in rows)
    assert rows[0]["answer"] == ["num", 12.0] and rows[0]["new_tokens"] == 5
    assert all(n == gen.MAX_NEW_TOKENS == 512 for _, n in engine.calls)


def test_committed_agreement_reads_the_committed_evaluation():
    path = REPO / "results/evals/run_a_r16-adapter_val_image_only_cot_0shot_bf16/predictions.jsonl"
    preds = [json.loads(line) for line in open(path, encoding="utf-8")]
    rows = [{"id": p["id"], "answer": list(gen.parsed_answer(p["raw"]))} for p in preds]
    got = M.committed_agreement(rows, path)
    assert got["n"] == got["agree"] == len(preds) > 100
    assert M.committed_agreement(rows, REPO / "no/such/file.jsonl") is None
