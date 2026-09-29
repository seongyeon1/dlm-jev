"""API contract tests with the model replaced by a stub reader."""

from typing import Any

import pytest
from fastapi.testclient import TestClient

from jevlike import server
from jevlike.engine import to_answer
from jevlike.schema import DecideRequest, DecideResponse, Usage


class StubReader:
    """Returns a fixed 70/30 split for every question."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    def decide(self, request: DecideRequest) -> DecideResponse:
        answers = {}
        for qid, q in request.questions.items():
            k = 2 if q.type == "noul" else len(getattr(q, "options", None) or q.levels)
            probs = [0.7] + [0.3 / (k - 1)] * (k - 1)
            answers[qid] = to_answer(q, probs)
        return DecideResponse(
            model="stub", answers=answers, usage=Usage(input_tokens=1, canvas_tokens=1, reads=1), timing_ms={}
        )


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    """Client with the stub reader loaded at startup."""
    monkeypatch.setattr(server, "DiffusionReader", StubReader)
    return TestClient(server.app)


def test_decide_returns_typed_answers(client: TestClient) -> None:
    """Every question id comes back with its own typed answer."""
    body = {
        "state": {"msg": "hi"},
        "questions": {
            "a": {"type": "noul", "instructions": "x"},
            "b": {"type": "choice", "instructions": "x", "options": ["p", "q", "r"]},
            "c": {"type": "score", "instructions": "x", "levels": ["lo", "hi"]},
        },
    }
    with client:
        res = client.post("/v1/decide", json=body)
    assert res.status_code == 200
    answers = res.json()["answers"]
    assert answers["a"]["noul"] == pytest.approx(0.7)
    assert answers["b"]["choice"] == "p"
    assert answers["c"]["level"] == "lo"


def test_off_schema_question_is_422(client: TestClient) -> None:
    """Unknown question types are rejected before touching the model."""
    with client:
        res = client.post("/v1/decide", json={"state": "s", "questions": {"a": {"type": "text", "instructions": "x"}}})
    assert res.status_code == 422
