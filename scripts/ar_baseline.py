"""Step 4: an autoregressive baseline on the same prompts, labels and items.

The AR model reads each question's label distribution at the next-token position after
"<qid>:" (restricted to the same single-token labels), then appends its own argmax
answer and moves on — the usual way an AR model answers several questions in order.
Accuracy/NLL are compared with the diffusion reader's stored raw reads on the same items;
latency is measured interleaved in one process with both models loaded.

    uv run python scripts/ar_baseline.py --model mlx-community/Qwen3-4B-4bit
"""

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import mlx.core as mx
import numpy as np
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache

sys.path.insert(0, str(Path(__file__).resolve().parent))
from benchmark import RUNS, tasks  # noqa: E402

from dlm_jev.calibration import Metrics, evaluate  # noqa: E402
from dlm_jev.canvas import SYSTEM_PROMPT, _slot_tokens, labels_for, render_state, render_user_parts  # noqa: E402
from dlm_jev.engine import DiffusionReader  # noqa: E402
from dlm_jev.schema import DecideRequest  # noqa: E402
from dlm_jev.synth import QUESTIONS, make_dataset  # noqa: E402


def render_state_first(request: DecideRequest) -> str:
    """Same content as the diffusion prompt, state before questions: the AR model's better order.

    Qwen3-4B on 40 SST-2 items: question-first 20/40 (always "no"), state-first 31/40.
    """
    head, tail = render_user_parts(request.state, dict(request.questions))
    state = render_state(request.state)
    return f"STATE:\n{state}\n\n" + head.replace("\n\nSTATE:\n", "\n\n") + tail[len(state) :]


class ARReader:
    """Label-restricted next-token reads from an AR chat model, one question after another."""

    def __init__(self, model_id: str) -> None:
        self.model, self.tokenizer = load(model_id)

    def _encode(self, text: str) -> List[int]:
        return list(self.tokenizer.encode(text, add_special_tokens=False))

    def read(self, request: DecideRequest) -> Tuple[Dict[str, List[float]], float]:
        """Label log-probabilities per question id, and wall time in ms."""
        t0 = time.perf_counter()
        questions = dict(request.questions)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": render_state_first(request)},
        ]
        prompt = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False, enable_thinking=False
        )
        cache = make_prompt_cache(self.model)
        pending = self._encode(prompt)
        out: Dict[str, List[float]] = {}
        for qid, q in questions.items():
            prefix, label_ids = _slot_tokens(self.tokenizer, qid, labels_for(q))
            logits = self.model(mx.array([pending + prefix]), cache=cache)[0, -1].astype(mx.float32)
            label = logits[mx.array(label_ids)]
            logp = label - mx.logsumexp(label)
            mx.eval(logp)
            out[qid] = np.array(logp, dtype=np.float64).tolist()
            pending = [label_ids[int(np.argmax(out[qid]))], *self._encode("\n")]
        return out, (time.perf_counter() - t0) * 1e3


def _fmt(m: Metrics) -> str:
    return f"acc={m.accuracy:.3f} nll={m.nll:.3f} ece={m.ece:.3f}"


def main() -> None:
    """Accuracy on stored items, then an interleaved latency A/B."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="mlx-community/Qwen3-4B-4bit")
    parser.add_argument("--public-per-task", type=int, default=100)
    parser.add_argument("--latency-n", type=int, default=30)
    args = parser.parse_args()
    ar = ARReader(args.model)
    report: Dict[str, Any] = {"model": args.model, "public": {}, "tickets": {}}

    base = {json.loads(line)["uid"]: json.loads(line) for line in (RUNS / "reads.jsonl").read_text().splitlines()}
    seen: Dict[str, int] = {}
    pub: Dict[str, List[tuple]] = {}
    for task, uid, request, target in tasks(200, 0):
        if seen.get(task, 0) >= args.public_per_task or uid not in base:
            continue
        seen[task] = seen.get(task, 0) + 1
        logp, _ = ar.read(request)
        pub.setdefault(task, []).append((base[uid]["logits"], logp["q"], target))
    for task, rs in pub.items():
        y = [r[2] for r in rs]
        d, a = evaluate([r[0] for r in rs], y), evaluate([r[1] for r in rs], y)
        print(f"{task:12s} diffusion {_fmt(d)} | AR {_fmt(a)}")
        report["public"][task] = {"diffusion": d.__dict__, "ar": a.__dict__}

    dif = {json.loads(line)["i"]: json.loads(line) for line in (RUNS / "reads-tickets.jsonl").read_text().splitlines()}
    data = make_dataset(600, 1)
    ar_rows = []
    for i in range(300, 600):
        logp, _ = ar.read(data[i][0])
        ar_rows.append((i, logp))
    for q in QUESTIONS:
        y = [data[i][1][q] for i, _ in ar_rows]
        d = evaluate([dif[i]["logits"][q] for i, _ in ar_rows], y)
        a = evaluate([lp[q] for _, lp in ar_rows], y)
        print(f"{'ticket:' + q:12s} diffusion {_fmt(d)} | AR {_fmt(a)}")
        report["tickets"][q] = {"diffusion": d.__dict__, "ar": a.__dict__}

    reader = DiffusionReader()
    lat: Dict[str, List[float]] = {"diffusion": [], "ar": []}
    for k in range(args.latency_n * 2):
        request = data[300 + k // 2][0]
        if k % 2 == 0:
            lat["diffusion"].append(reader.read(request).timing_ms["total"])
        else:
            lat["ar"].append(ar.read(request)[1])
    for name, v in lat.items():
        print(f"latency {name:9s} p50={statistics.median(v):.0f}ms p90={np.percentile(v, 90):.0f}ms (4 questions)")
    report["latency_ms"] = {
        k: {"p50": statistics.median(v), "p90": float(np.percentile(v, 90))} for k, v in lat.items()
    }
    (RUNS / "report-ar-baseline.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
