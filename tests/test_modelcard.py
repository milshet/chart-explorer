"""Task 4.6: the model card is generated, current, honest about gaps, and quotes the committed numbers."""
import json
import re

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
from serve import modelcard as MC  # noqa: E402
from serve.quantise import mcnemar_exact  # noqa: E402

SECTIONS = ["## Model", "## Intended use", "## Training data provenance", "## Licences", "## Evaluation", "## Failure modes",
            "## Refusal behaviour", "## Deployment", "## Limitations", "## Not measured"]


@pytest.fixture(scope="module")
def card():
    return MC.build()


def body(text):
    return text.split("## Not measured")[0]


def overlay(tmp_path, extra):
    """A root that sees the repository's committed files read-only, plus `extra` {relative path: content}."""
    for top in ("data", "blind_labelling"):
        (tmp_path / top).symlink_to(REPO / top)
    (tmp_path / "results").mkdir()
    for sub in (REPO / "results").iterdir():
        (tmp_path / "results" / sub.name).symlink_to(sub)
    for rel, content in extra.items():
        p = tmp_path / rel
        d = tmp_path
        for part in Path(rel).parent.parts:  # never write through a symlink into the repository
            d = d / part
            if d.is_symlink():
                real = d.resolve()
                d.unlink()
                d.mkdir()
                for child in real.iterdir():
                    (d / child.name).symlink_to(child)
            d.mkdir(exist_ok=True)
        if p.is_symlink():  # a committed file of the same name: replace the link, never write through it
            p.unlink()
        assert not any(q.is_symlink() for q in [p, *p.parents] if tmp_path in q.parents or q == tmp_path)
        p.write_text(content if isinstance(content, str) else json.dumps(content), encoding="utf-8")
    return tmp_path


# --- the committed card ----------------------------------------------------------------------------------------

def test_the_committed_card_is_a_fresh_build(card):
    assert (REPO / MC.CARD).read_text(encoding="utf-8") == card[0], "run scripts/serve/modelcard.py --write"


def test_check_mode_agrees():
    assert MC.main(["--check"]) == 0


def test_every_section_is_present_in_order(card):
    positions = [card[0].index(s) for s in SECTIONS]
    assert positions == sorted(positions)


def test_every_not_measured_in_the_body_is_listed_as_a_gap(card):
    text, gaps = card
    assert body(text).count(MC.NM) == len(gaps)
    assert f"{len(gaps)} item(s) in this card are NOT MEASURED" in text
    assert all(f"- {g}" in text for g in gaps)


def test_the_gaps_today_are_the_audit_kappa_and_the_container(card):
    gaps = card[1]
    assert len(gaps) == 2  # the judge kappa (3.7) landed on 2026-11-23
    assert any("3.0i" in g for g in gaps) and any("4.4" in g for g in gaps) and not any("3.7" in g for g in gaps)


# --- numbers come from the files -----------------------------------------------------------------------------------

def test_the_headline_accuracy_is_the_committed_test_summary(card):
    s = json.loads((REPO / "results/evals/run_a_r16-adapter_test_image_only_cot_0shot_bf16/summary.json").read_text())
    assert f"**{100 * s['overall']['relaxed_accuracy']:.1f}%**" in card[0] and s["overall"]["n"] == 180


def test_comparisons_are_paired_mcnemar_from_the_predictions(card):
    got = MC.paired(MC.Card(REPO), "run_a_r16-adapter_test_image_only_cot_0shot_bf16",
                    "base_test_image_only_answer_only_2shot_4bit")
    a, b, n, only_a, only_b, p = got
    assert n == 180 and p == mcnemar_exact(only_a, only_b)
    assert f"| base model, 2-shot | {b}/180" in card[0] and f"| {only_a} / {only_b} | {p:.2g} |" in card[0]


def test_the_merge_and_quantisation_verdicts_are_quoted(card):
    m = json.loads((REPO / "results/serve/merge_run_a_r16.json").read_text())
    q = json.loads((REPO / "results/serve/quantise_run_a_r16.json").read_text())
    assert f"Merge status **{m['status']}**" in card[0]
    assert "4-bit quantisation was rejected" in card[0] and f"{100 * q['accuracy']['drop']:.2f} points" in card[0]


def test_the_red_team_result_and_the_stated_then_answered_pattern(card):
    r = json.loads((REPO / "results/redteam/run_a_r16.json").read_text())
    assert f"Resisted **{r['summary']['resisted']}/21**" in card[0]
    n = sum(a["outcome"] == "failed_to_abstain" for a in r["attempts"])
    assert re.search(rf"In \d of the {n} failures to abstain the rationale states", card[0])


def test_the_selection_and_labelling_counts_are_computed(card):
    st = json.loads((REPO / "data/v2.0/stats.json").read_text())["splits"]["train"]
    assert f"| train | {st['selected']:,} | {st['kept']:,} | {100 * st['kept'] / st['selected']:.1f}% |" in card[0]
    assert "pilot 38/40" in card[0] and "29/30" in card[0]


def test_serving_facts_say_what_was_actually_deployed(card):
    text = card[0]
    assert "Not vLLM" in text and "bf16" in text and "`degraded`" in text
    assert "The card describes the bf16 model" in text


def test_intended_use_names_what_it_is_not_for(card):
    use = card[0].split("## Intended use")[1].split("##")[0]
    assert "**For:**" in use and "**Not for:**" in use and "financial, investment or policy decisions" in use


# --- missing inputs render NOT MEASURED, present ones replace it ----------------------------------------------------

def test_an_empty_repository_renders_every_section_as_not_measured(tmp_path):
    text, gaps = MC.build(tmp_path)
    assert all(s in text for s in SECTIONS) and len(gaps) >= 8
    assert body(text).count(MC.NM) == len(gaps)


def test_the_audit_kappa_appears_once_its_output_is_committed(tmp_path):
    root = overlay(tmp_path, {"results/audit/kappa.txt": "answer_correct n= 40  Fleiss kappa=0.71\n"})
    text, gaps = MC.build(root)
    assert "Fleiss kappa=0.71" in text and not any("3.0i" in g for g in gaps)


def test_the_judge_kappa_appears_once_agreement_is_computed(tmp_path):
    agreement = {"correctness": {"human_vs_human": [30, 0.9, 0.78], "judge_vs_grader_1": [30, 0.8, 0.5],
                                 "judge_vs_grader_2": [30, 1.0, None]}}
    root = overlay(tmp_path, {"results/judge/run_a_r16/agreement.json": agreement})
    text, gaps = MC.build(root)
    assert "| correctness | κ 0.78 | κ 0.50 | κ undefined (raw 1.00) |" in text and not any("3.7" in g for g in gaps)


def test_the_committed_judge_kappa_is_in_the_card(card):
    a = json.loads((REPO / "results/judge/run_a_r16/agreement.json").read_text())
    assert f"| grounding | κ {a['grounding']['human_vs_human'][2]:.2f} |" in card[0]


def test_a_container_smoke_log_replaces_the_container_gap(tmp_path):
    root = overlay(tmp_path, {"results/serve/container_smoke.json": {"summary": "healthy, 1 request answered"}})
    text, gaps = MC.build(root)
    assert "built and served: healthy, 1 request answered" in text and not any("4.4" in g for g in gaps)


def test_without_the_red_team_log_the_card_says_so(tmp_path):
    for top in ("data", "blind_labelling"):
        (tmp_path / top).symlink_to(REPO / top)
    (tmp_path / "results").mkdir()
    for sub in (REPO / "results").iterdir():
        if sub.name != "redteam":
            (tmp_path / "results" / sub.name).symlink_to(sub)
    text, gaps = MC.build(tmp_path)
    assert "Adversarial charts: NOT MEASURED" in text and any("4.5" in g for g in gaps)


# --- licences ---------------------------------------------------------------------------------------------------------

def test_the_card_states_the_base_models_non_commercial_licence_in_intended_use(card):
    use = card[0].split("## Intended use")[1].split("## ")[0]
    assert "**Research and evaluation only.**" in use and "Qwen Research License" in use


def test_licences_are_recorded_for_the_revisions_actually_used(card):
    lic = json.loads((REPO / "docs/licences.json").read_text())
    merge = json.loads((REPO / "results/serve/merge_run_a_r16.json").read_text())
    served = next(x for x in lic["components"] if x["role"] == "served")
    data = next(x for x in lic["components"] if x["kind"] == "dataset")
    assert served["revision"] == merge["base_check"]["base_revision"]
    assert data["revision"][:12] in (REPO / "data/v2.0/README.md").read_text()
    assert "differs from the revision used" not in card[0]


def test_every_component_and_obligation_is_rendered(card):
    lic = json.loads((REPO / "docs/licences.json").read_text())
    section = card[0].split("## Licences")[1].split("## Evaluation")[0]
    for x in lic["components"]:
        assert x["name"] in section and all(o in section for o in x["obligations"])
    assert "Built with Qwen" in section and "GPL-3.0" in section


def test_a_licence_recorded_for_another_revision_is_flagged(tmp_path):
    lic = json.loads((REPO / "docs/licences.json").read_text())
    lic["components"][0]["revision"] = "0" * 40
    root = overlay(tmp_path, {"docs/licences.json": lic})
    assert "differs from the revision used" in MC.build(root)[0]


def test_without_a_licence_record_the_card_says_so(tmp_path):
    root = overlay(tmp_path, {})
    text, gaps = MC.build(root)
    assert any("docs/licences.json" in g for g in gaps) and "Licences: NOT MEASURED" in text


def test_built_with_qwen_is_displayed_at_the_top_and_in_the_readme(card):
    top = card[0].split("## Model")[0]
    assert "**Built with Qwen.**" in top
    assert "Built with Qwen" in (REPO / "README.md").read_text()
