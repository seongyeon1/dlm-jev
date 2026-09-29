"""Typed answer conversion and schema tests."""

import pytest
from pydantic import ValidationError

from jevlike.engine import to_answer
from jevlike.schema import ChoiceQuestion, DecideRequest, NoulQuestion, ScoreQuestion


def test_noul_is_probability_of_yes() -> None:
    """noul = P(first label = yes)."""
    ans = to_answer(NoulQuestion(instructions="x"), [0.8, 0.2])
    assert ans.noul == pytest.approx(0.8)


def test_score_expected_value() -> None:
    """Score returns argmax level and expectation over level indices."""
    ans = to_answer(ScoreQuestion(instructions="x", levels=["a", "b", "c"]), [0.2, 0.3, 0.5])
    assert (ans.score, ans.level) == (2, "c")
    assert ans.expected == pytest.approx(1.3)


def test_choice_maps_options() -> None:
    """Choice returns the option text, not the letter."""
    ans = to_answer(ChoiceQuestion(instructions="x", options=["p", "q"]), [0.1, 0.9])
    assert ans.choice == "q"


def test_request_rejects_unknown_type_and_bad_ids() -> None:
    """Off-schema questions fail validation."""
    with pytest.raises(ValidationError):
        DecideRequest(state="s", questions={"q": {"type": "free_text", "instructions": "x"}})
    with pytest.raises(ValidationError):
        DecideRequest(state="s", questions={"1bad id": {"type": "noul", "instructions": "x"}})
