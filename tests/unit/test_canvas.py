"""Canvas layout tests."""

import pytest

from dlm_jev.canvas import (
    CHOICE_LABELS,
    MIN_CANVAS_LENGTH,
    NOUL_LABELS,
    build_canvas,
    render_user_message,
    render_user_parts,
)
from dlm_jev.schema import ChoiceQuestion, NoulQuestion, ScoreQuestion


def test_slots_sit_after_prefix_and_read_label_tokens(tokenizer) -> None:
    """Each slot must follow 'qid:' and be read over exactly that question's label tokens."""
    # Given
    questions = {
        "spam": NoulQuestion(instructions="Is this spam?"),
        "topic": ChoiceQuestion(instructions="Topic?", options=["sports", "tech", "world"]),
    }
    # When
    canvas = build_canvas(tokenizer, questions, noise_id=0, end_ids=[1])
    # Then
    assert [s.question_id for s in canvas.slots] == ["spam", "topic"]
    assert canvas.slots[0].token_ids == tuple(tokenizer.vocab[" " + lab] for lab in NOUL_LABELS)
    assert canvas.slots[1].labels == CHOICE_LABELS[:3]
    for slot in canvas.slots:
        assert canvas.token_ids[slot.position] == 0
        assert canvas.token_ids[slot.position - 1] == tokenizer.vocab[":"]
    assert canvas.token_ids[canvas.slots[-1].position + 1] == 1


def test_canvas_is_padded_to_min_length(tokenizer) -> None:
    """Regression: reads on an answer-length canvas collapse to chance, so the canvas is padded."""
    # Given
    questions = {"q": NoulQuestion(instructions="x")}
    # When
    canvas = build_canvas(tokenizer, questions, noise_id=0, end_ids=[1], lead_ids=[7, 8], pad_id=3)
    # Then
    assert len(canvas.token_ids) == MIN_CANVAS_LENGTH
    assert canvas.token_ids[:2] == (7, 8)
    assert canvas.token_ids[-1] == 3


def test_prompt_lists_the_same_labels_the_slot_reads(tokenizer) -> None:
    """The prompt must advertise exactly the labels the canvas reads."""
    # Given
    questions = {"q1": ScoreQuestion(instructions="Urgency?", levels=["low", "mid", "high"])}
    # When
    text = render_user_message("server down", questions)
    canvas = build_canvas(tokenizer, questions, noise_id=0, end_ids=[])
    # Then
    for label in canvas.slots[0].labels:
        assert f"  {label}: " in text
    assert "q1: <one of 0|1|2>" in text


def test_multi_token_label_is_rejected() -> None:
    """A tokenizer that splits labels must fail loudly instead of reading a wrong token."""

    class CharTokenizer:
        def encode(self, text: str, add_special_tokens: bool = False) -> list:
            return [ord(c) for c in text] + ([0] if text.endswith("yes") else [])

    with pytest.raises(ValueError):
        build_canvas(CharTokenizer(), {"q": NoulQuestion(instructions="x")}, noise_id=0, end_ids=[])


def test_question_head_is_shared_across_states() -> None:
    """Requests with the same questions must share the head exactly (it is the prefix-cache key)."""
    # Given
    questions = {"q": ChoiceQuestion(instructions="Topic?", options=["a", "b"])}
    # When
    head1, tail1 = render_user_parts("first state", questions)
    head2, tail2 = render_user_parts({"other": "state"}, questions)
    # Then
    assert head1 == head2
    assert "first state" not in head1 and "first state" in tail1
    assert tail1 != tail2
