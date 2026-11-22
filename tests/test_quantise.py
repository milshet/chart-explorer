"""Task 4.2 on the laptop: the target plan, the export check, calibration disjointness and the verdict.

The checkpoint is described by a synthetic safetensors header with Qwen2.5-VL-3B's module layout (36 decoder layers
of 7 projections, a vision tower of attention and MLP blocks, a merger, lm_head), which is all the two module checks
read. Calibration is checked against the real evaluation sets in the repository.
"""
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
from serve import generation as gen  # noqa: E402
from serve import quantise as Q  # noqa: E402

DEC = [f"model.language_model.layers.{i}.{p}" for i in range(36) for p in
       ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
        "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj")]
VIS = [f"model.visual.blocks.{i}.{p}" for i in range(32) for p in
       ("attn.qkv", "attn.proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj")] + \
      ["model.visual.merger.mlp.0", "model.visual.merger.mlp.2"]
OTHER = ["lm_head", "model.language_model.embed_tokens"]


def header(quantised=(), vision_dtype="BF16", drop=()):
    """{tensor: entry} like a safetensors header; modules in `quantised` carry GPTQ tensors instead of .weight."""
    h = {}
    for m in DEC + VIS + OTHER:
        if m in drop:
            continue
        if m in quantised:
            for s in ("qweight", "qzeros", "scales", "g_idx"):
                h[f"{m}.{s}"] = {"dtype": "I32", "shape": [256, 2048]}
        else:
            h[f"{m}.weight"] = {"dtype": vision_dtype if ".visual." in m else "BF16", "shape": [2048, 2048]}
        if ".visual." not in m and m != "lm_head":
            h[f"{m}.bias"] = {"dtype": "BF16", "shape": [2048]}
    h["model.language_model.norm.weight"] = {"dtype": "BF16", "shape": [2048]}
    return h


# --- targets ----------------------------------------------------------------------------------------------------

def test_plan_quantises_exactly_the_decoder_projections():
    plan = Q.plan_targets(Q.linear_modules(header()))
    assert sorted(plan["quantise"]) == sorted(DEC) and len(plan["quantise"]) == Q.EXPECTED_DECODER_PROJ == 252
    assert plan["keep"] == {"vision tower and merger": len(VIS), "lm_head": 1}
    assert plan["other_linear_kept"] == ["model.language_model.embed_tokens"]


def test_linear_modules_ignores_1d_tensors():
    mods = Q.linear_modules(header())
    assert "model.language_model.norm" not in mods and "lm_head" in mods


def test_a_skip_pattern_that_matches_nothing_raises():
    renamed = {n.replace("visual", "vision_tower"): v for n, v in header().items()}
    with pytest.raises(Q.QuantError, match="matches no module"):
        Q.plan_targets(Q.linear_modules(renamed))


def test_a_pattern_that_misses_the_vision_tower_raises_before_anything_runs():
    keep = {"vision tower and merger": r"(^|\.)visual\.merger\.", "lm_head": r"(^|\.)lm_head$"}
    with pytest.raises(Q.QuantError, match="vision-tower module is covered"):
        Q.plan_targets(Q.linear_modules(header()), keep=keep)


def test_vision_mlp_projections_named_like_decoder_ones_are_never_targets():
    """The vision blocks have gate/up/down_proj too: the decoder pattern must not pick them up."""
    plan = Q.plan_targets(Q.linear_modules(header()))
    assert not any(".visual." in m for m in plan["quantise"])


def test_a_decoder_count_other_than_252_raises():
    with pytest.raises(Q.QuantError, match="251 decoder projections"):
        Q.plan_targets(Q.linear_modules(header(drop=DEC[:1])))


def test_checkpoint_tensors_reads_every_shard(tmp_path):
    sys.path.insert(0, str(REPO / "tests"))
    from test_merge import write_safetensors
    write_safetensors(tmp_path / "model-00001-of-00002.safetensors", {"a.weight": ("BF16", (2, 2))})
    write_safetensors(tmp_path / "model-00002-of-00002.safetensors", {"b.weight": ("BF16", (2, 2))})
    assert set(Q.checkpoint_tensors(tmp_path)) == {"a.weight", "b.weight"}
    with pytest.raises(Q.QuantError, match="no .safetensors"):
        Q.checkpoint_tensors(tmp_path / "nothing")


# --- the export ---------------------------------------------------------------------------------------------------

def test_a_correct_export_passes():
    plan = Q.plan_targets(Q.linear_modules(header()))
    got = Q.inspect_export(header(quantised=DEC), plan)
    assert got["quantised_modules"] == 252 and got["vision_weight_dtypes"] == ["BF16"] and got["tensor_bytes"] > 0


def test_a_decoder_projection_left_unquantised_raises():
    plan = Q.plan_targets(Q.linear_modules(header()))
    with pytest.raises(Q.QuantError, match="1 decoder projection"):
        Q.inspect_export(header(quantised=DEC[1:]), plan)


def test_a_quantised_vision_module_raises():
    plan = Q.plan_targets(Q.linear_modules(header()))
    with pytest.raises(Q.QuantError, match="must stay bf16"):
        Q.inspect_export(header(quantised=DEC + VIS[:1]), plan)


def test_a_quantised_lm_head_raises():
    plan = Q.plan_targets(Q.linear_modules(header()))
    with pytest.raises(Q.QuantError, match="must stay bf16"):
        Q.inspect_export(header(quantised=DEC + ["lm_head"]), plan)


def test_a_vision_tower_saved_in_fp16_is_allowed_and_recorded():
    plan = Q.plan_targets(Q.linear_modules(header()))
    assert Q.inspect_export(header(quantised=DEC, vision_dtype="F16"), plan)["vision_weight_dtypes"] == ["F16"]


def test_a_vision_tower_stored_as_integers_raises():
    plan = Q.plan_targets(Q.linear_modules(header()))
    with pytest.raises(Q.QuantError, match="not unquantised floating point"):
        Q.inspect_export(header(quantised=DEC, vision_dtype="I8"), plan)


def test_an_export_without_a_vision_tower_raises():
    plan = Q.plan_targets(Q.linear_modules(header()))
    h = {n: v for n, v in header(quantised=DEC).items() if ".visual." not in n}
    with pytest.raises(Q.QuantError, match="no vision-tower weights"):
        Q.inspect_export(h, plan)


def test_calibration_conversation_is_the_training_conversation_with_training_pixels(train_rows, evals):
    import smoke_test as st
    r = Q.calibration_rows(train_rows, evals)[0]
    conv = Q.calibration_conversation(r)
    assert [m["role"] for m in conv] == ["system", "user", "assistant"]
    assert conv[0] == r["messages"][0] and conv[2]["content"] == r["messages"][2]["content"]
    img, text = conv[1]["content"]
    assert text == r["messages"][1]["content"][1] and img["max_pixels"] == st.MAX_PIXELS
    assert img["image"].tobytes() == st.load_image(REPO / "data/v2.0", r["image"]).tobytes()
    assert r["messages"][1]["content"][0]["image"] == r["image"]  # the dataset row itself is not modified


def test_gptq_settings_are_fixed():
    assert Q.quantize_config_kwargs() == {"bits": 4, "group_size": 128, "sym": True, "desc_act": False}
    assert set(Q.BACKENDS) == {"gptqmodel"}


def test_compressed_tensors_naming_counts_as_quantised():
    plan = Q.plan_targets(Q.linear_modules(header()))
    h = {n: v for n, v in header().items() if not any(n == f"{m}.weight" for m in DEC)}
    for m in DEC:
        h[f"{m}.weight_packed"] = {"dtype": "I32", "shape": [2048, 256]}
        h[f"{m}.weight_scale"] = {"dtype": "BF16", "shape": [2048, 16]}
    assert Q.inspect_export(h, plan)["quantised_modules"] == 252


# --- calibration ----------------------------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def evals():
    return Q.eval_fingerprints()


@pytest.fixture(scope="module")
def train_rows():
    return Q.read_jsonl(REPO / "data/v2.0/image_only/train.jsonl")


def test_eval_fingerprints_cover_every_evaluation_set(evals):
    test_ids = {r["id"] for r in Q.read_jsonl(REPO / "data/v2.0/image_only/test.jsonl")}
    eval_c = {r["id"] for r in Q.read_jsonl(REPO / "data/eval_c/image_only/eval_c.jsonl")}
    assert test_ids <= evals["ids"] and eval_c <= evals["ids"]
    assert len(evals["images"]) == 249  # distinct chart images across val, test, Eval-C and the RAG sets


def test_calibration_is_drawn_from_train_deterministically_and_disjoint(train_rows, evals):
    a = Q.calibration_rows(train_rows, evals)
    assert len(a) == Q.CALIB_N == 128 and all(r["id"].startswith("train-") for r in a)
    assert [r["id"] for r in Q.calibration_rows(list(reversed(train_rows)), evals)] == [r["id"] for r in a]


@pytest.mark.parametrize("field", ["ids", "charts"])
def test_a_calibration_row_in_an_evaluation_set_raises(train_rows, evals, field):
    first = Q.calibration_rows(train_rows, evals)[0]
    leaked = {**evals, field: evals[field] | {first["id"] if field == "ids" else first["chart_id"]}}
    with pytest.raises(Q.QuantError, match="overlap the evaluation sets"):
        Q.calibration_rows(train_rows, leaked)


def test_identical_image_bytes_under_another_name_raise(train_rows, evals):
    first = Q.calibration_rows(train_rows, evals)[0]
    digest = Q.image_digest(REPO / "data/v2.0" / first["image"])
    with pytest.raises(Q.QuantError, match="image bytes identical"):
        Q.calibration_rows(train_rows, {**evals, "images": evals["images"] | {digest}})


def test_no_evaluation_set_found_raises():
    with pytest.raises(Q.QuantError, match="no evaluation set"):
        Q.eval_fingerprints(sets=[("data/nowhere", "test")])


# --- the verdict ------------------------------------------------------------------------------------------------

VALID = {"evidence": [], "rationale": "r", "answer_type": "number", "unit": None, "answer": 1}


def row(i, correct, valid=True, answer=1):
    raw = json.dumps({**VALID, "answer": answer}) if valid else "not json"
    return {"id": f"t{i}", "correct": correct, "raw": raw, "style": "cot", "answer": list(gen.parsed_answer(raw)),
            "new_tokens": 70, "seconds": 1.2}


def test_verdict_accepts_within_tolerance():
    ref = [row(i, i < 150) for i in range(180)]
    q = [row(i, i < 147) for i in range(180)]  # 3 lost of 180 = 1.67 points
    v = Q.verdict(ref, q)
    assert v["verdict"] == "ACCEPT" and v["accuracy"]["drop"] == round(3 / 180, 4)
    assert v["discordant"] == {"lost": 3, "gained": 0, "mcnemar_p": 0.25}


def test_verdict_rejects_past_the_accuracy_tolerance():
    ref = [row(i, i < 150) for i in range(180)]
    q = [row(i, i < 146) for i in range(180)]  # 4 lost = 2.2 points
    assert Q.verdict(ref, q)["verdict"] == "REJECT"


def test_verdict_rejects_on_lost_contract_validity_even_at_equal_accuracy():
    ref = [row(i, False) for i in range(180)]
    q = [row(i, False, valid=i > 1) for i in range(180)]
    v = Q.verdict(ref, q)
    assert v["verdict"] == "REJECT" and v["contract_valid"] == {"bf16": 180, "quantised": 178}


def test_gains_offset_losses_in_the_accuracy_delta():
    ref = [row(i, i < 100) for i in range(180)]
    q = [row(i, 5 <= i < 105) for i in range(180)]
    v = Q.verdict(ref, q)
    assert v["accuracy"]["drop"] == 0 and v["discordant"]["lost"] == v["discordant"]["gained"] == 5
    assert v["verdict"] == "ACCEPT"


def test_verdict_needs_identical_rows():
    with pytest.raises(ValueError):
        Q.verdict([row(0, True)], [row(1, True)])


def test_mcnemar_exact():
    assert Q.mcnemar_exact(0, 0) == 1.0
    assert Q.mcnemar_exact(5, 0) == 0.0625
    assert Q.mcnemar_exact(10, 3) == pytest.approx(0.0923, abs=1e-4)  # the 3.0j vision-arm comparison in the report


def test_tolerances_are_fixed_constants():
    assert (Q.BITS, Q.GROUP_SIZE, Q.CALIB_N, Q.MAX_ACCURACY_DROP, Q.MAX_VALID_DROP, Q.HEADLINE_SPLIT) == \
        (4, 128, 128, 0.02, 1, "test")


def test_a_backend_failure_is_a_recorded_downgrade_with_its_exception():
    try:
        raise ImportError("No module named 'gptqmodel'")
    except ImportError as e:
        d = Q.downgrade("quantise", e)
    assert d["verdict"] == "REJECT" and d["status"] == "DOWNGRADE_BACKEND_FAILED"
    assert "ImportError: No module named 'gptqmodel'" == d["exception"] and "gptqmodel" in d["traceback"]
    assert "serve the bf16 merge" in d["fallback"]
