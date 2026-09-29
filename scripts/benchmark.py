"""Read public labelled datasets through the slot reader, fit per-type temperatures, report calibration.

Tasks (one per question type):
  noul   SST-2          "the review is positive"
  choice AG News        4 topics
  score  Yelp full      1-5 stars

Raw reads are cached to ``runs/reads.jsonl`` so fitting can be re-run without the model.

    uv run python scripts/benchmark.py --per-task 200
"""

import argparse
import json
import random
from pathlib import Path
from typing import Dict, Iterator, List, Tuple

from datasets import load_dataset

from jevlike.calibration import Calibrator, Metrics, evaluate
from jevlike.schema import ChoiceQuestion, DecideRequest, NoulQuestion, ScoreQuestion

RUNS = Path(__file__).resolve().parent.parent / "runs"
AG_TOPICS = ["World", "Sports", "Business", "Science/Technology"]
STARS = ["1 star", "2 stars", "3 stars", "4 stars", "5 stars"]


def tasks(per_task: int, seed: int) -> Iterator[Tuple[str, str, DecideRequest, int]]:
    """Yield (task, uid, request, target_index)."""
    rng = random.Random(seed)
    sst = load_dataset("stanfordnlp/sst2", split="validation")
    for i in rng.sample(range(len(sst)), per_task):
        row = sst[i]
        q = NoulQuestion(instructions="The review expresses a positive opinion of the movie.")
        target = 0 if row["label"] == 1 else 1
        yield "sst2", f"sst2-{i}", DecideRequest(state=row["sentence"], questions={"q": q}), target
    ag = load_dataset("fancyzhx/ag_news", split="test")
    for i in rng.sample(range(len(ag)), per_task):
        row = ag[i]
        q = ChoiceQuestion(instructions="Which section of a newspaper does this article belong to?", options=AG_TOPICS)
        yield "ag_news", f"ag-{i}", DecideRequest(state=row["text"], questions={"q": q}), int(row["label"])
    yelp = load_dataset("Yelp/yelp_review_full", split="test")
    for i in rng.sample(range(len(yelp)), per_task):
        row = yelp[i]
        q = ScoreQuestion(instructions="How many stars did the reviewer give?", levels=STARS)
        yield "yelp", f"yelp-{i}", DecideRequest(state=row["text"][:2000], questions={"q": q}), int(row["label"])


def collect(per_task: int, seed: int, out: Path) -> None:
    """Run the reader over all tasks, appending to ``out`` (resumable by uid)."""
    from jevlike.engine import DiffusionReader

    done = {json.loads(line)["uid"] for line in out.read_text().splitlines()} if out.exists() else set()
    reader = DiffusionReader()
    with out.open("a") as f:
        for task, uid, request, target in tasks(per_task, seed):
            if uid in done:
                continue
            res = reader.read(request)
            row = {
                "task": task,
                "uid": uid,
                "type": request.questions["q"].type,
                "logits": list(res.slots[0].logits),
                "target": target,
                "timing_ms": res.timing_ms,
                "prompt_tokens": res.prompt_tokens,
                "cached_tokens": res.cached_tokens,
                "reads": res.reads,
            }
            f.write(json.dumps(row) + "\n")
            f.flush()
            print(task, uid, f"{res.timing_ms['total']:.0f}ms", "reads", res.reads)


def _fmt(m: Metrics) -> str:
    return f"n={m.n:4d} acc={m.accuracy:.3f} nll={m.nll:.3f} brier={m.brier:.3f} ece={m.ece:.3f}"


def report(path: Path, seed: int) -> Dict[str, object]:
    """Split reads 50/50 per task, fit T on one half, report both halves' metrics on the other."""
    by_task: Dict[str, List[dict]] = {}
    for line in path.read_text().splitlines():
        r = json.loads(line)
        by_task.setdefault(r["task"], []).append(r)
    rng = random.Random(seed)
    fit_rows: Dict[str, List[tuple]] = {}
    test_rows: Dict[str, List[dict]] = {}
    for task, rs in by_task.items():
        rng.shuffle(rs)
        half = len(rs) // 2
        test_rows[task] = rs[:half]
        for r in rs[half:]:
            fit_rows.setdefault(r["type"], []).append((r["logits"], r["target"]))
    calibrator = Calibrator.fit(fit_rows, name="temperature-v1")
    calibrator.save(RUNS / "calibration.json")
    summary: Dict[str, object] = {"temperatures": calibrator.temperatures, "tasks": {}}
    print("temperatures:", {k: round(v, 3) for k, v in calibrator.temperatures.items()})
    for task, rs in sorted(test_rows.items()):
        logits, targets = [r["logits"] for r in rs], [r["target"] for r in rs]
        t = calibrator.temperature(rs[0]["type"])
        raw, cal = evaluate(logits, targets), evaluate(logits, targets, t)
        lat = sorted(r["timing_ms"]["total"] for r in rs)
        print(f"{task:8s} raw  {_fmt(raw)}")
        print(f"{task:8s} cal  {_fmt(cal)}   p50={lat[len(lat) // 2]:.0f}ms")
        summary["tasks"][task] = {"raw": raw.__dict__, "calibrated": cal.__dict__, "p50_ms": lat[len(lat) // 2]}
    (RUNS / "report.json").write_text(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    """CLI."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-task", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args()
    RUNS.mkdir(exist_ok=True)
    reads = RUNS / "reads.jsonl"
    if not args.report_only:
        collect(args.per_task, args.seed, reads)
    report(reads, args.seed)


if __name__ == "__main__":
    main()
