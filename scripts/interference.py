"""Do other questions in the same request change a question's answer?

For each labelled item, the task question is read three ways on the same state:
  alone      — the only question
  first      — followed by two unrelated questions
  last       — preceded by the same two questions
and the label distributions are compared (paired, same items).

    uv run python scripts/interference.py --per-task 100
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from benchmark import RUNS, tasks  # noqa: E402

from dlm_jev.calibration import evaluate  # noqa: E402
from dlm_jev.engine import DiffusionReader  # noqa: E402
from dlm_jev.schema import DecideRequest, NoulQuestion, ScoreQuestion  # noqa: E402

# Unrelated to every task's label, answerable from any text.
DISTRACTORS = {
    "mentions_number": NoulQuestion(instructions="The text contains at least one number or digit."),
    "formality": ScoreQuestion(
        instructions="How formal is the writing style?", levels=["very casual", "casual", "neutral", "formal"]
    ),
}


def variants(request: DecideRequest) -> Dict[str, DecideRequest]:
    """The task question alone, first in a bundle, and last in a bundle."""
    q = request.questions["q"]
    return {
        "alone": request,
        "first": DecideRequest(state=request.state, questions={"q": q, **DISTRACTORS}),
        "last": DecideRequest(state=request.state, questions={**DISTRACTORS, "q": q}),
    }


def main() -> None:
    """Collect reads for all variants, then report paired differences against ``alone``."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-task", type=int, default=100)
    parser.add_argument("--seed", type=int, default=2)
    args = parser.parse_args()
    reader = DiffusionReader()
    rows: List[dict] = []
    for task, uid, request, target in tasks(args.per_task, args.seed):
        row = {"task": task, "uid": uid, "target": target}
        for name, req in variants(request).items():
            res = reader.read(req)
            row[name] = list(next(s for s in res.slots if s.question_id == "q").logits)
        rows.append(row)
    (RUNS / "interference.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    print(f"{'task':8s} {'variant':6s} {'acc':>6s} {'nll':>6s} {'flips':>6s} {'mean|dp|':>9s} {'max|dp|':>8s}")
    for task in ("sst2", "ag_news", "yelp"):
        rs = [r for r in rows if r["task"] == task]
        y = [r["target"] for r in rs]
        base = [np.exp(r["alone"]) for r in rs]
        for name in ("alone", "first", "last"):
            lg = [r[name] for r in rs]
            m = evaluate(lg, y)
            p = [np.exp(x) for x in lg]
            flips = sum(int(np.argmax(a) != np.argmax(b)) for a, b in zip(base, p, strict=True))
            dp = [float(np.abs(a - b).max()) for a, b in zip(base, p, strict=True)]
            print(f"{task:8s} {name:6s} {m.accuracy:6.3f} {m.nll:6.3f} {flips:6d} {np.mean(dp):9.3f} {np.max(dp):8.3f}")


if __name__ == "__main__":
    main()
