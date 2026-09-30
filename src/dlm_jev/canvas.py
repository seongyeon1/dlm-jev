"""Prompt and canvas layout: where each question's answer slot sits in the decoder canvas.

The canvas is the model's response written in advance, e.g.::

    q1: <slot>\\n
    q2: <slot>\\n
    <end>

Template tokens are pinned; each slot starts as random noise and one decoder
pass gives its distribution. Only the label tokens of that slot are read, so an
answer cannot leave the schema.
"""

import json
import string
from dataclasses import dataclass
from typing import Dict, List, Protocol, Sequence, Tuple, Union

from dlm_jev.schema import ChoiceQuestion, NoulQuestion, Question, ScoreQuestion

# Single source for labels: the prompt text and the slot read both use these.
NOUL_LABELS: Tuple[str, str] = ("yes", "no")
CHOICE_LABELS: Tuple[str, ...] = tuple(string.ascii_uppercase)
SCORE_LABELS: Tuple[str, ...] = tuple(string.digits)

# DiffusionGemma's default minimum canvas; shorter canvases read at chance (see docs/DESIGN.md §6).
MIN_CANVAS_LENGTH = 64

# (joiner after "qid", format of the label) tried in order until every label is one token.
_SLOT_FORMS: Tuple[Tuple[str, str], ...] = ((":", " {}"), (": ", "{}"))

SYSTEM_PROMPT = (
    "You are a decision engine. Read the STATE, then answer every question with "
    "exactly one label from that question's label list. Do not explain."
)


class Tokenizer(Protocol):
    """Minimal tokenizer surface used for canvas layout."""

    def encode(self, text: str, add_special_tokens: bool = ...) -> List[int]: ...


@dataclass(frozen=True)
class Slot:
    """One answer slot: canvas position and the label token ids it is read over."""

    question_id: str
    position: int
    labels: Tuple[str, ...]
    token_ids: Tuple[int, ...]


@dataclass(frozen=True)
class Canvas:
    """Pinned canvas tokens plus the slots to read."""

    token_ids: Tuple[int, ...]
    slots: Tuple[Slot, ...]


def labels_for(question: Question) -> Tuple[str, ...]:
    """Return the label strings a question is answered with."""
    if isinstance(question, NoulQuestion):
        return NOUL_LABELS
    if isinstance(question, ChoiceQuestion):
        return CHOICE_LABELS[: len(question.options)]
    return SCORE_LABELS[: len(question.levels)]


def render_state(state: Union[str, Dict[str, object], List[str]]) -> str:
    """Render the state block as text."""
    if isinstance(state, str):
        return state
    if isinstance(state, dict):
        return json.dumps(state, ensure_ascii=False, indent=1)
    return "\n".join(f"[{i}] {item}" for i, item in enumerate(state, start=1))


def _render_question(question_id: str, question: Question) -> str:
    labels = labels_for(question)
    if isinstance(question, NoulQuestion):
        body = f"Labels: {labels[0]} = the statement is true, {labels[1]} = it is false"
    elif isinstance(question, ChoiceQuestion):
        body = "\n".join(f"  {lab}: {opt}" for lab, opt in zip(labels, question.options, strict=True))
        body = f"Labels:\n{body}"
    elif isinstance(question, ScoreQuestion):
        body = "\n".join(f"  {lab}: {lvl}" for lab, lvl in zip(labels, question.levels, strict=True))
        body = f"Labels (ordered, lowest first):\n{body}"
    return f"[{question_id}] ({question.type}) {question.instructions}\n{body}"


def render_user_parts(
    state: Union[str, Dict[str, object], List[str]],
    questions: Dict[str, Question],
) -> Tuple[str, str]:
    """Render the user turn as (question head, state tail).

    Questions come first so that requests sharing a question set share a prompt
    prefix whose encoder cache can be reused; only the state tail is prefilled.
    """
    blocks = "\n\n".join(_render_question(qid, q) for qid, q in questions.items())
    # The label set is repeated next to the answer format: the question block sits before the state.
    answer_format = "\n".join(f"{qid}: <one of {'|'.join(labels_for(q))}>" for qid, q in questions.items())
    head = f"QUESTIONS:\n{blocks}\n\nSTATE:\n"
    tail = (
        f"{render_state(state)}\n\n"
        f"Answer every question above about this STATE, one line per question, in this exact format:\n"
        f"{answer_format}"
    )
    return head, tail


def render_user_message(
    state: Union[str, Dict[str, object], List[str]],
    questions: Dict[str, Question],
) -> str:
    """Render the full user turn: questions with their labels, the state, and the answer format."""
    head, tail = render_user_parts(state, questions)
    return head + tail


def _slot_tokens(tokenizer: Tokenizer, question_id: str, labels: Sequence[str]) -> Tuple[List[int], Tuple[int, ...]]:
    """Find a prefix encoding after which every label is exactly one token."""
    for joiner, label_format in _SLOT_FORMS:
        prefix_text = f"{question_id}{joiner}"
        prefix = tokenizer.encode(prefix_text, add_special_tokens=False)
        ids: List[int] = []
        for label in labels:
            full = tokenizer.encode(prefix_text + label_format.format(label), add_special_tokens=False)
            if full[: len(prefix)] != prefix or len(full) != len(prefix) + 1:
                break
            ids.append(full[-1])
        else:
            if len(set(ids)) == len(ids):
                return prefix, tuple(ids)
    raise ValueError(f"labels {list(labels)} are not single tokens after '{question_id}:'")


def build_canvas(
    tokenizer: Tokenizer,
    questions: Dict[str, Question],
    noise_id: int,
    end_ids: Sequence[int],
    lead_ids: Sequence[int] = (),
    pad_id: int = 0,
    min_length: int = MIN_CANVAS_LENGTH,
) -> Canvas:
    """Lay out the canvas; slot positions hold ``noise_id`` as a placeholder.

    ``lead_ids`` is whatever the model writes before its answer (e.g. an empty
    thought channel). The canvas is padded to ``min_length`` with ``pad_id``: the
    model is trained on canvases of at least that length and reads collapse to
    chance on a canvas that is only as long as the answer.
    """
    newline = tokenizer.encode("\n", add_special_tokens=False)
    tokens: List[int] = list(lead_ids)
    slots: List[Slot] = []
    for i, (question_id, question) in enumerate(questions.items()):
        labels = labels_for(question)
        prefix, label_ids = _slot_tokens(tokenizer, question_id, labels)
        tokens.extend(prefix)
        slots.append(Slot(question_id, len(tokens), labels, label_ids))
        tokens.append(noise_id)
        if i < len(questions) - 1:
            tokens.extend(newline)
    tokens.extend(end_ids)
    tokens.extend([pad_id] * (min_length - len(tokens)))
    return Canvas(tuple(tokens), tuple(slots))
