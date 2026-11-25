"""The 12-page report (docs/REPORT.md) is hand-written prose, so its key numbers are recomputed here from the
committed results: if a result changes and the report is not updated, this fails instead of the report drifting."""
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
from serve import modelcard as MC  # noqa: E402

REPORT = (REPO / "docs/REPORT.md").read_text(encoding="utf-8")
CARD = MC.Card(REPO)
TEST = "run_a_r16-adapter_test_image_only_cot_0shot_bf16"
VAL = "run_a_r16-adapter_val_image_only_cot_0shot_bf16"


def read(rel):
    return json.loads((REPO / rel).read_text(encoding="utf-8"))


@pytest.mark.parametrize("other, label", [
    ("base_test_image_only_answer_only_2shot_4bit", "base"),
    ("ablation_answer_only-adapter_test_image_only_answer_only_0shot_bf16", "ablation"),
    ("run_b_r64-adapter_test_image_only_cot_0shot_bf16", "run_b"),
    ("run_a_r16_vision-adapter_test_image_only_cot_0shot_bf16", "vision"),
])
def test_test_split_comparisons(other, label):
    a, b, n, only_a, only_b, p = MC.paired(CARD, TEST, other)
    assert f"{b} ({100 * b / n:.1f}%)" in REPORT and f"{only_a} / {only_b}, p = {p:.2g}" in REPORT.replace(
        "53 only Run A / 5 only base", "53 / 5"), label


@pytest.mark.parametrize("other", ["run_b_r64-adapter_val_image_only_cot_0shot_bf16",
                                   "ablation_answer_only-adapter_val_image_only_answer_only_0shot_bf16",
                                   "run_a_r16_vision-adapter_val_image_only_cot_0shot_bf16"])
def test_val_selection_table(other):
    a, b, n, only_a, only_b, p = MC.paired(CARD, VAL, other)
    assert f"| {b} ({100 * b / n:.1f}%) |" in REPORT and f"{only_b} / {only_a}" in REPORT and f"p = {p:.2g}" in REPORT


def test_served_model_numbers():
    m = read("results/serve/merge_run_a_r16.json")
    h = read("results/serve/run_a_r16_http.json")["summary"]
    q = read("results/serve/quantise_run_a_r16.json")
    assert f"0 of {m['weight_check']['elements_changed_by_merge']:,}" in REPORT
    assert f"{h['e2e_s']['p50']} s" in REPORT and f"{h['e2e_s']['p95']} s" in REPORT
    assert f"{100 * q['accuracy']['quantised']:.1f}% against {100 * q['accuracy']['bf16']:.1f}%" in REPORT
    assert f"{100 * q['accuracy']['drop']:.2f}-point drop" in REPORT


def test_red_team_and_abstention_numbers():
    r = read("results/redteam/run_a_r16.json")["summary"]
    assert f"resisted {r['resisted']} of 21" in REPORT
    for fam, counts in r["by_family"].items():
        assert f"| {counts.get('resisted', 0)}/3 |" in REPORT
    e = read(
        "results/evals/run_a_r16-adapter_eval_c_image_only_cot_0shot_bf16/summary.json")["overall"]["abstention"]
    assert f"| Run A (served) | {round(e['true_abstention'] * 50)} |" in REPORT


def test_red_team_failures_are_described_as_counted():
    o = read("results/redteam/run_a_r16.json")["summary"]["outcomes"]
    assert (o["failed_to_abstain"], o["wrong"], o["misled"]) == (5, 1, 0)
    assert "The other five failures are failures to abstain" in REPORT
    assert "Every failure is a failure to abstain" not in REPORT


def test_judge_kappa_table():
    a = read("results/judge/run_a_r16/agreement.json")
    for axis, v in a.items():
        row = f"| {axis.capitalize()} | {v['human_vs_human'][2]:.2f} | {v['judge_vs_grader_1'][2]:.2f} | " \
              f"{v['judge_vs_grader_2'][2]:.2f} |"
        assert row in REPORT, row


def test_dataset_numbers():
    st = read("data/v2.0/stats.json")["splits"]
    for split in ("train", "val"):
        s = st[split]
        assert f"| {split} | {s['selected']:,} | {s['kept']:,} | {100 * s['kept'] / s['selected']:.1f}% |" in REPORT


def test_the_report_states_every_deviation_and_gap():
    for needle in ("not vLLM", "Eval-B", "4-bit", "never been built or run", "not carried out", "Qwen Research License",
                   "Built with Qwen", "one request at a time"):
        assert needle in REPORT, needle


def test_the_clean_clone_figures_match_the_committed_check():
    r = read("results/release/clean_clone.json")
    passed = r["checks"]["tests_summary"].split(" passed")[0].split()[-1]
    assert f"{r['checks']['manifest_files']:,}" in REPORT and f"{passed} tests passing inside the clone" in REPORT
