"""Does averaging re-reads beat a single read? Same items, same prefill, 1 vs k reads.

uv run python scripts/reread_value.py --per-task 100 --k 4
"""

import argparse
import json
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from benchmark import RUNS, tasks  # noqa: E402

from dlm_jev.calibration import evaluate  # noqa: E402
from dlm_jev.engine import DiffusionReader  # noqa: E402


def main() -> None:
    """Collect k reads per item, then compare read-1 vs mean-of-k on the same items."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-task", type=int, default=100)
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()
    reader = DiffusionReader()
    rows = []
    for task, uid, request, target in tasks(args.per_task, args.seed):
        if task == "sst2":
            continue
        canvas = reader.layout(request)
        cache = reader.model.diffusion_prefill_cache(
            mx.array([reader.prompt_ids(request)[0]]),
            attention_mask=None,
            cache=reader.model.make_cache(),
            pixel_values=None,
            mm_token_type_ids=None,
            prefill_step_size=None,
            chunk_prefill=False,
        )
        reads = [r[0] for r in reader._read_batch(canvas, cache, args.k)]
        rows.append({"task": task, "uid": uid, "target": target, "reads": [r.tolist() for r in reads]})
    (RUNS / "reread_value.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    for task in ("ag_news", "yelp"):
        rs = [r for r in rows if r["task"] == task]
        y = [r["target"] for r in rs]
        one = [np.log(np.maximum(r["reads"][0], 1e-12)) for r in rs]
        avg = [np.log(np.maximum(np.mean(r["reads"], axis=0), 1e-12)) for r in rs]
        flips = sum(int(np.argmax(r["reads"][0]) != np.argmax(np.mean(r["reads"], axis=0))) for r in rs)
        for name, lg in (("1 read", one), (f"mean of {args.k}", avg)):
            m = evaluate(lg, y)
            print(f"{task:8s} {name:10s} acc={m.accuracy:.3f} nll={m.nll:.3f} brier={m.brier:.3f} ece={m.ece:.3f}")
        print(f"{task:8s} argmax changed by averaging: {flips}/{len(rs)}")


if __name__ == "__main__":
    main()
