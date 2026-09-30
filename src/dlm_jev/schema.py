"""Request/response types for Jev-style typed decisions."""

from typing import Annotated, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator

QuestionId = Annotated[str, Field(pattern=r"^[A-Za-z][A-Za-z0-9_]{0,31}$")]


class NoulQuestion(BaseModel):
    """Yes/no statement evaluated as P(yes)."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["noul"] = "noul"
    instructions: str = Field(min_length=1)


class ChoiceQuestion(BaseModel):
    """Pick one option out of 2..26."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["choice"] = "choice"
    instructions: str = Field(min_length=1)
    options: List[str] = Field(min_length=2, max_length=26)

    @field_validator("options")
    @classmethod
    def _unique_options(cls, options: List[str]) -> List[str]:
        if len(set(options)) != len(options):
            raise ValueError("options must be unique")
        return options


class ScoreQuestion(BaseModel):
    """Rate the state on ordered levels (lowest first), 2..10 levels."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["score"] = "score"
    instructions: str = Field(min_length=1)
    levels: List[str] = Field(min_length=2, max_length=10)


Question = Annotated[Union[NoulQuestion, ChoiceQuestion, ScoreQuestion], Field(discriminator="type")]


class DecideRequest(BaseModel):
    """A state plus typed questions answered in one read."""

    model_config = ConfigDict(extra="forbid")

    state: Union[str, Dict[str, object], List[str]]
    questions: Dict[QuestionId, Question] = Field(min_length=1, max_length=32)
    calibrated: bool = True


class NoulAnswer(BaseModel):
    """Answer to a noul question."""

    type: Literal["noul"] = "noul"
    noul: float
    confidence: float


class ChoiceAnswer(BaseModel):
    """Answer to a choice question."""

    type: Literal["choice"] = "choice"
    choice: str
    probabilities: Dict[str, float]
    confidence: float


class ScoreAnswer(BaseModel):
    """Answer to a score question; ``score`` is the most likely level index."""

    type: Literal["score"] = "score"
    score: int
    level: str
    expected: float
    probabilities: Dict[str, float]
    confidence: float


Answer = Union[NoulAnswer, ChoiceAnswer, ScoreAnswer]


class Usage(BaseModel):
    """Token and pass accounting for one request."""

    input_tokens: int
    cached_tokens: int = 0
    canvas_tokens: int
    reads: int


class DecideResponse(BaseModel):
    """Answers keyed by question id."""

    model: str
    answers: Dict[str, Answer]
    usage: Usage
    timing_ms: Dict[str, float]
    calibration: Optional[str] = None
