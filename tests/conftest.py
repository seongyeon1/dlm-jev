"""Shared fixtures: a tiny word-level tokenizer so canvas layout is testable without the model."""

import re
from typing import Dict, List

import pytest


class WordTokenizer:
    """Splits into words, ':' and newlines; a leading space is part of the next word."""

    def __init__(self) -> None:
        self.vocab: Dict[str, int] = {}

    def encode(self, text: str, add_special_tokens: bool = False) -> List[int]:
        pieces = re.findall(r" ?[A-Za-z0-9_]+|:|\n| ", text)
        return [self.vocab.setdefault(p, len(self.vocab) + 10) for p in pieces]


@pytest.fixture
def tokenizer() -> WordTokenizer:
    """Fresh word tokenizer."""
    return WordTokenizer()
