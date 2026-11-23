"""Task 4.8 on the laptop: the demo's requests, in demo order, through the real app with a scripted model; what each
step must show; and the generated script with its "do not claim" list."""
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
from serve import client  # noqa: E402
from serve import demo as D  # noqa: E402
from serve import generation as gen  # noqa: E402
from serve import service as svc  # noqa: E402

MANIFEST = {"status": "VERIFIED", "run": "run_a_r16", "adapter_sha256": "9d9b3d3a" * 8,
            "merged_files_sha256": {"model.safetensors": "ab" * 32}}


def answer(value, rationale="read from the chart", kind="number"):
    return json.dumps({"evidence": ["bar labels"], "rationale": rationale, "answer_type": kind, "unit": None,
                       "answer": value})


class ByQuestion:
    """A scripted model that answers each demo question the way the step expects (or the way given in `override`)."""

    def __init__(self, override=None):
        injection = D.spec("redteam-09-prompt_injection")
        caption = D.spec("redteam-15-caption_only_answer")
        self.replies = {
            D.item("data/v2.0/image_only/val.jsonl", "val-two_col_24100-651")["question"]: answer(38),
            D.item("data/eval_c/image_only/eval_c.jsonl", "eval_c-002-series_absent")["question"]:
                answer(None, "The chart shows revenue by category; it has no net income.", "unanswerable"),
            injection["question"]: answer(injection["answer"]),
            caption["question"]: answer(470, "The chart does not show 2022; management expects 470."),
            **(override or {})}

    def generate(self, messages, images, max_new_tokens=512):
        text = messages[1]["content"][1]["text"]
        q = text.split("\n\n")[0].removeprefix("Question: ")
        return gen.Generation(self.replies[q], prompt_tokens=700, new_tokens=60, truncated=False, seconds=1.1)


@pytest.fixture
def endpoint(monkeypatch):
    """The real FastAPI app around a Service with a scripted model; client.get/post routed to it."""
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    from serve import app as app_module

    def make(engine):
        http = TestClient(app_module.create_app(svc.Service(engine, MANIFEST, "cot")))
        monkeypatch.setattr(client, "get", lambda url, path, timeout=10: http.get(path).json())

        def post(url, body, timeout=120):
            r = http.post("/v1/answer", content=body)
            return r.status_code, r.json()
        monkeypatch.setattr(client, "post", post)
        return http
    return make


# --- the steps -------------------------------------------------------------------------------------------------------

def test_the_demo_has_seven_steps_in_time_order():
    minutes = [tuple(map(int, s["minute"].split(":"))) for s in D.STEPS]
    assert len(D.STEPS) == 7 and minutes == sorted(minutes) and minutes[-1] < (7, 0)
    assert [s["name"] for s in D.STEPS] == ["health", "answer", "abstain", "injection", "caption_failure",
                                            "refused_context", "metrics"]


def test_every_post_body_is_accepted_or_refused_exactly_as_the_step_expects():
    for step in D.STEPS:
        if step["kind"] == "get":
            continue
        body, _ = D.request_for(step)
        if step["expect"]["status"] == 200:
            question, image = svc.parse_request(body)
            assert question and image.size[0] * image.size[1] <= svc.MAX_PIXELS
        else:
            with pytest.raises(svc.Refusal) as r:
                svc.parse_request(body)
            assert r.value.code == step["expect"]["code"] and r.value.status == step["expect"]["status"]


def test_the_demo_items_are_ones_the_served_model_got_right_when_measured():
    preds = {json.loads(l)["id"]: json.loads(l) for l in
             (REPO / "results/evals/run_a_r16-adapter_val_image_only_cot_0shot_bf16/predictions.jsonl").read_text().splitlines()}
    evc = {json.loads(l)["id"]: json.loads(l) for l in
           (REPO / "results/evals/run_a_r16-adapter_eval_c_image_only_cot_0shot_bf16/predictions.jsonl").read_text().splitlines()}
    red = {a["id"]: a for a in json.loads((REPO / "results/redteam/run_a_r16.json").read_text())["attempts"]}
    assert preds["val-two_col_24100-651"]["correct"] and evc["eval_c-002-series_absent"]["abstained"]
    assert red["redteam-09-prompt_injection"]["outcome"] == "resisted"
    assert red["redteam-15-caption_only_answer"]["outcome"] == "failed_to_abstain"  # the failure shown on purpose


# --- the sequence, in order, through the app ------------------------------------------------------------------------

def test_the_whole_demo_passes_in_order_against_the_app(endpoint):
    endpoint(ByQuestion())
    rows = D.run_steps("http://demo")
    assert [r["name"] for r in rows] == [s["name"] for s in D.STEPS]
    assert all(r["passed"] for r in rows), [r for r in rows if not r["passed"]]
    by = {r["name"]: r["observed"] for r in rows}
    assert by["answer"]["correct"] and by["abstain"]["abstained"] and not by["injection"]["token_in_output"]
    assert by["refused_context"]["code"] == "caller_text_refused" and by["metrics"]["counters.degraded"] == 0


def test_a_wrong_answer_fails_its_step(endpoint):
    q = D.item("data/v2.0/image_only/val.jsonl", "val-two_col_24100-651")["question"]
    endpoint(ByQuestion({q: answer(41)}))
    rows = {r["name"]: r for r in D.run_steps("http://demo")}
    assert not rows["answer"]["passed"] and rows["abstain"]["passed"]


def test_obeying_the_injection_fails_its_step(endpoint):
    s = D.spec("redteam-09-prompt_injection")
    endpoint(ByQuestion({s["question"]: answer(s["injection_token"], kind="text")}))
    rows = {r["name"]: r for r in D.run_steps("http://demo")}
    assert not rows["injection"]["passed"] and rows["injection"]["observed"]["token_in_output"]


def test_a_degraded_fallback_fails_its_step_and_shows_in_the_metrics_step(endpoint):
    q = D.item("data/eval_c/image_only/eval_c.jsonl", "eval_c-002-series_absent")["question"]
    endpoint(ByQuestion({q: "not json"}))
    rows = {r["name"]: r for r in D.run_steps("http://demo")}
    assert not rows["abstain"]["passed"]  # the fallback is null too, but it is not the model's abstention
    assert not rows["metrics"]["passed"] and rows["metrics"]["observed"]["counters.degraded"] == 1


# --- the script ---------------------------------------------------------------------------------------------------------

def smoke(passed=True):
    rows = [{"name": s["name"], "seconds": 1.0, "passed": passed or s["name"] != "abstain",
             "observed": {"status": s["expect"]["status"], "answer": 38}} for s in D.STEPS]
    return {"slurm_job_id": "160", "git_commit": "abcdef0", "steps": rows}


def test_the_script_marks_a_step_that_failed_the_smoke_test():
    text = D.render(smoke(passed=False))
    assert "6/7 steps passed" in text
    abstain = text.split("### 3:00 abstain")[1].split("###")[0]
    assert "NOT TO BE SHOWN" in abstain and "NOT TO BE SHOWN" not in text.split("### 1:30 answer")[1].split("###")[0]


def test_without_a_smoke_run_the_script_says_so():
    text = D.render(None)
    assert "Smoke test in demo order: NOT MEASURED" in text and text.count("smoke: NOT MEASURED") == 7


def test_the_do_not_claim_list_comes_from_the_runs_own_gaps():
    items = D.do_not_claim()
    joined = " | ".join(items)
    for needle in ("vLLM", "vision-tower", "quantised deployment", "concurrent load", "reliable abstention",
                   "Eval-B", "(3.0i)", "(3.7)", "(task 4.4)"):
        assert needle in joined, needle
    assert len(items) == len(set(items))


def test_the_results_part_quotes_committed_numbers():
    lines = D.headline()
    assert "151/180" in lines[0] and "103/180" in lines[0] and "15/21" in lines[1] and "2.22-point" in lines[2]
    assert all("NOT MEASURED" not in l for l in lines)


def test_the_committed_script_is_current():
    path = REPO / "docs/demo_script.md"
    if not path.is_file():
        pytest.skip("docs/demo_script.md is written after the smoke run")
    smoke_file = D.SMOKE
    data = json.loads(smoke_file.read_text()) if smoke_file.is_file() else None
    assert path.read_text(encoding="utf-8") == D.render(data), "run scripts/serve/demo.py --write"
