import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

import scoring as S  # noqa: E402

TABLE = "Title: Demo\nTable:\n| Country | Value |\n|---|---|\n| Italy | 22 |"


def make_item(answer, answer_type="number", question="What is the value?"):
    target = {"answer": answer, "answer_type": answer_type, "unit": None, "evidence": [], "rationale": "r"}
    return {
        "id": "demo",
        "answer_type": answer_type,
        "question": question,
        "input_text": TABLE,
        "messages": [{}, {}, {"role": "assistant", "content": json.dumps(target)}],
    }


def output(**fields):
    base = {"answer": 1, "answer_type": "number", "unit": None, "evidence": [], "rationale": "r"}
    base.update(fields)
    return json.dumps(base)


# Item 1: years
@pytest.mark.parametrize("question", [
    "In which year the market share is highest?",
    "Which year has the highest wage?",
    "When did the blue bar reach its peak?",
    "When the military expenditure reached a value higher than 0.8%?",
])
def test_nearby_year_fails_on_year_question(question):
    assert not S.relaxed_match(2015, 2019, "number", question)
    assert S.relaxed_match(2019, 2019, "number", question)


def test_value_that_looks_like_a_year_keeps_tolerance():
    assert S.relaxed_match(1950, 2000, "number", "What is the revenue in Italy?")


def test_ordinary_number_keeps_tolerance():
    assert S.relaxed_match(93.2, 91.1, "number", "What is the average share?")
    assert not S.relaxed_match(100, 91.1, "number", "What is the average share?")


# Item 2: lists
@pytest.mark.parametrize("pred", [
    "Australia and Italy",
    "Italy & Australia",
    "[Italy, Australia]",
    "Australia,Italy",
    ["Italy", "Australia"],
])
def test_list_answers_match_in_any_form(pred):
    assert S.relaxed_match(pred, "[Australia, Italy]", "text", "Which two countries?")


def test_wrong_list_fails():
    assert not S.relaxed_match("2003, 2010", "[2003,2009]", "text", "What 2 years did it peak?")
    assert not S.relaxed_match("Australia", "[Australia, Italy]", "text")
    assert not S.relaxed_match("Australia, Italy, Spain", "[Australia, Italy]", "text")


def test_list_of_years_is_exact():
    assert S.relaxed_match("2009 and 2003", "[2003,2009]", "text", "What 2 years did it peak?")
    assert not S.relaxed_match("2004, 2009", "[2003,2009]", "text", "What 2 years did it peak?")


def test_list_with_numeric_item_uses_tolerance():
    assert S.relaxed_match("Italy, 22.5", "[Italy , 22]", "text")


# Item 3: refusals
def test_missing_answer_key_is_not_a_refusal():
    item = make_item(None, "unanswerable")
    assert not S.score(item, '{"rationale": "cannot tell"}')["correct"]


def test_explicit_null_is_a_refusal():
    item = make_item(None, "unanswerable")
    assert S.score(item, output(answer=None, answer_type="unanswerable"))["correct"]


def test_answering_an_unanswerable_item_fails():
    item = make_item(None, "unanswerable")
    assert not S.score(item, output(answer=5))["correct"]


# Item 4: escaped quotes
@pytest.mark.parametrize("rationale", [
    'one "quoted" word',
    'odd " quote',
    "a backslash \\ here",
    'brace } and quote " inside',
])
def test_extract_json_handles_escapes(rationale):
    text = json.dumps({"answer": 1, "rationale": rationale})
    assert S.extract_json(text) == {"answer": 1, "rationale": rationale}


def test_extract_json_handles_fences_and_incomplete_output():
    assert S.extract_json('```json\n{"answer": 1}\n```') == {"answer": 1}
    assert S.extract_json('{"answer": 1, "rationale": "cut o') is None
    assert S.extract_json("no json here") is None


# Item 5: bad answer_type
@pytest.mark.parametrize("bad", [["number"], {"a": 1}, 5, None])
def test_bad_answer_type_is_invalid_not_a_crash(bad):
    result = S.score(make_item(1), output(answer_type=bad))
    assert result["schema_valid"] is False
    assert result["correct"] is True


# Items 8 and 9: length and truncation
def test_word_counts():
    result = S.score(make_item("Italy", "text"), output(answer="Italy and Spain", answer_type="text"))
    assert (result["gold_words"], result["pred_words"]) == (1, 3)
    assert S.word_count("[2003,2009]") == 2
    assert S.word_count(None) == 0


def test_truncated_flag_is_recorded():
    assert S.score(make_item(1), output(), truncated=True)["truncated"] is True
    assert S.score(make_item(1), output())["truncated"] is False


def test_evidence_is_still_checked():
    good = "Row=Italy, Col=Value (22)"
    result = S.score(make_item(22), output(answer=22, evidence=[good, "free text"]))
    assert (result["evidence_n"], result["evidence_ok"]) == (2, 1)
