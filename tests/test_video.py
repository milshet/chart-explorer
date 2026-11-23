"""The demo video's generated segments say only what the committed results say, and the voice reads them sensibly.
Rendering needs ffmpeg and macOS `say`, so it is not run here; the text that goes into it is."""
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
from serve import demo as D  # noqa: E402
from serve import video as V  # noqa: E402


@pytest.mark.parametrize("text, said", [
    ("p = 3.5e-11", "p equals 3.5 times ten to the minus 11"),
    ("151/180 correct", "151 of 180 correct"),
    ("p50 1.224 s, HTTP adds 3 ms", "p50 1.224 seconds, HTTP adds 3 milliseconds"),
    ("a 2.22-point drop", "a 2.22 point drop"),
    ("the audit κ (3.0i)", "the audit kappa"),
    ("run with docker compose on a GPU (task 4.4)", "run with docker compose on a GPU"),
    ("81.1% vs 83.3%", "81.1% versus 83.3%"),
    ("served with `transformers.generate`", "served with transformers.generate"),
])
def test_spoken(text, said):
    assert V.spoken(text) == said


def test_the_slides_carry_the_committed_numbers_and_claims():
    seg = V.segments()
    assert set(seg) == {"intro", "results", "close"}
    assert seg["results"][0]["lines"] == D.headline()
    shown = [l for s in seg["close"] for l in s["lines"]]
    assert shown == [V.short(c) for c in D.do_not_claim()]
    assert "1,033" in seg["intro"][0]["lines"][0] and "research and evaluation use only" in seg["intro"][0]["lines"][2]


def test_every_slide_is_narrated_and_fits_on_screen():
    for slides in V.segments().values():
        for s in slides:
            assert s["narration"].strip()
            rows = sum(len(V.textwrap.wrap(l, 70)) for l in s["lines"])
            assert 300 + rows * 54 + len(s["lines"]) * 24 < V.H - 100, s["title"]  # clear of the footer


def test_video_output_is_never_committed():
    assert "build/" in (REPO / ".gitignore").read_text()


def test_cues_keep_their_marks_unless_the_previous_line_runs_long():
    cues = [{"at": 0.5}, {"at": 5.0}, {"at": 6.0}]
    starts, end = V.place(cues, [3.0, 2.0, 1.0])
    assert starts == [0.5, 5.0, 7.3] and end == 8.3   # the third waits for the second, plus a 0.3 s gap


def test_the_committed_voiceover_cues_are_in_order_and_quote_what_the_recording_shows():
    import json
    spec = json.loads((REPO / "docs/demo_voiceover.json").read_text())
    ats = [c["at"] for c in spec["cues"]]
    assert ats == sorted(ats) and {c["step"] for c in spec["cues"]} >= {s["name"] for s in D.STEPS}
    said = " ".join(c["text"] for c in spec["cues"])
    smoke = {r["name"]: r["observed"] for r in json.loads((REPO / "results/demo/smoke.json").read_text())["steps"]}
    assert smoke["answer"]["answer"] == 38 and "thirty-eight" in said
    assert smoke["injection"]["answer"] == 60 and "sixty" in said
    assert smoke["caption_failure"]["answer"] == 470 and "four hundred and seventy" in said
    assert smoke["refused_context"]["status"] == 400 and "four hundred error" in said
