"""Support-ticket triage: read synthetic tickets, fit per-question temperatures, report calibration.

All four questions are asked in one request (the production form), and calibration
is fitted in that same form.

    uv run python scripts/domain_eval.py --n 600
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from benchmark import RUNS  # noqa: E402

from jevlike.calibration import Calibrator, Metrics, evaluate  # noqa: E402
from jevlike.engine import DiffusionReader  # noqa: E402
from jevlike.synth import QUESTIONS, make_dataset  # noqa: E402


def collect(n: int, seed: int, out: Path) -> None:
    """Read every ticket once (all four questions in one pass); resumable by index."""
    done = {json.loads(line)["i"] for line in out.read_text().splitlines()} if out.exists() else set()
    reader = DiffusionReader()
    with out.open("a") as f:
        for i, (request, y, t) in enumerate(make_dataset(n, seed)):
            if i in done:
                continue
            res = reader.read(request)
            row = {
                "i": i,
                "labels": y,
                "attrs": t.__dict__,
                "logits": {s.question_id: list(s.logits) for s in res.slots},
                "timing_ms": res.timing_ms,
                "cached_tokens": res.cached_tokens,
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()


def _fmt(m: Metrics) -> str:
    return f"acc={m.accuracy:.3f} nll={m.nll:.3f} brier={m.brier:.3f} ece={m.ece:.3f}"


def report(path: Path) -> Dict[str, object]:
    """First half fits per-question T, second half is the test set."""
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows.sort(key=lambda r: r["i"])
    half = len(rows) // 2
    fit, test = rows[:half], rows[half:]
    samples = {f"q:{q}": [(r["logits"][q], r["labels"][q]) for r in fit] for q in QUESTIONS}
    per_q = Calibrator.fit(samples, name="support-tickets-v1")
    per_q.save(RUNS / "calibration-tickets.json")
    generic = Calibrator.load(RUNS / "calibration.json")  # public-data per-type fit, for comparison
    summary: Dict[str, object] = {"temperatures": per_q.temperatures, "questions": {}}
    print("per-question T:", {k: round(v, 2) for k, v in per_q.temperatures.items()})
    for q, question in QUESTIONS.items():
        lg, y = [r["logits"][q] for r in test], [r["labels"][q] for r in test]
        raw = evaluate(lg, y)
        gen = evaluate(lg, y, generic.temperature(question.type))
        dom = evaluate(lg, y, per_q.temperature(question.type, q))
        print(f"{q:12s} raw     {_fmt(raw)}")
        print(f"{q:12s} generic {_fmt(gen)}  (public-data T={generic.temperature(question.type):.2f})")
        print(f"{q:12s} domain  {_fmt(dom)}  (T={per_q.temperature(question.type, q):.2f})")
        summary["questions"][q] = {"raw": raw.__dict__, "generic": gen.__dict__, "domain": dom.__dict__}
    # hard cases on the test half
    slices = {
        "churn | decoy '취소'": ([r for r in test if r["attrs"]["decoy_cancel"] and not r["attrs"]["churn"]], "churn"),
        "churn | real intent": ([r for r in test if r["attrs"]["churn"]], "churn"),
        "category | billing words in non-billing": ([r for r in test if r["attrs"]["decoy_billing_words"]], "category"),
        "urgency | angry & low impact": (
            [r for r in test if r["attrs"]["tone"] == "angry" and r["labels"]["urgency"] == 0],
            "urgency",
        ),
    }
    for name, (rs, q) in slices.items():
        acc = np.mean([np.argmax(r["logits"][q]) == r["labels"][q] for r in rs]) if rs else float("nan")
        print(f"slice {name:42s} n={len(rs):3d} acc={acc:.3f}")
    lat = sorted(r["timing_ms"]["total"] for r in test)
    print(f"latency p50={lat[len(lat) // 2]:.0f}ms p90={lat[int(len(lat) * 0.9)]:.0f}ms (4 questions per request)")
    confusion: List[List[int]] = [[0] * len(QUESTIONS["urgency"].levels) for _ in QUESTIONS["urgency"].levels]
    for r in test:
        confusion[r["labels"]["urgency"]][int(np.argmax(r["logits"]["urgency"]))] += 1
    print("urgency confusion (row=true, col=pred):", confusion)
    (RUNS / "report-tickets.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


def main() -> None:
    """CLI."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=600)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args()
    RUNS.mkdir(exist_ok=True)
    reads = RUNS / "reads-tickets.jsonl"
    if not args.report_only:
        collect(args.n, args.seed, reads)
    report(reads)


if __name__ == "__main__":
    main()
