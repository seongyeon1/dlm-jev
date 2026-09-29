"""Latency profile: the smoke request (3 questions) plus a 1-question request, repeated.

uv run python scripts/latency.py --runs 10
"""

import argparse
import json
import statistics
from typing import Dict, List

from smoke import REQUEST

from jevlike.engine import DiffusionReader
from jevlike.schema import DecideRequest

SINGLE = {
    "state": "The pasta was cold and the waiter ignored us for twenty minutes.",
    "questions": {"positive": {"type": "noul", "instructions": "The review is positive."}},
}


def profile(reader: DiffusionReader, body: dict, runs: int) -> Dict[str, float]:
    """Median timings over ``runs`` after one warm-up call."""
    request = DecideRequest.model_validate(body)
    reader.decide(request)
    samples: List[Dict[str, float]] = []
    reads: List[int] = []
    for _ in range(runs):
        res = reader.decide(request)
        samples.append(res.timing_ms)
        reads.append(res.usage.reads)
    out = {k: round(statistics.median(s[k] for s in samples), 1) for k in samples[0]}
    out["reads_median"] = statistics.median(reads)
    return out


def main() -> None:
    """CLI."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=10)
    args = parser.parse_args()
    reader = DiffusionReader()
    result = {
        "three_questions": profile(reader, REQUEST, args.runs),
        "one_question": profile(reader, SINGLE, args.runs),
    }
    print(json.dumps(result))


if __name__ == "__main__":
    main()
