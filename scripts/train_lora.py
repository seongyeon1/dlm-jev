"""L2: LoRA fine-tune on synthetic support tickets, then compare against the base model.

Train on a disjoint generator seed; evaluate on the same test half as scripts/domain_eval.py
(seed 1, items 300..599), and check public-benchmark regression on 100 items per task.
The encoder cache is computed outside the gradient (stop-gradient through the prompt).

    uv run python scripts/train_lora.py --n-train 400
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Dict, List

import mlx.core as mx
import mlx.optimizers as optim
import numpy as np
from mlx import nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from benchmark import RUNS, tasks  # noqa: E402

from jevlike.calibration import Metrics, evaluate  # noqa: E402
from jevlike.engine import DiffusionReader  # noqa: E402
from jevlike.lora import apply_lora, save_adapter, slot_loss  # noqa: E402
from jevlike.synth import QUESTIONS, make_dataset  # noqa: E402


def train(reader: DiffusionReader, n_train: int, seed: int, lr: float, rank: int, scale: float) -> Path:
    """One epoch over ``n_train`` synthetic tickets, batch 1."""
    model = reader.model
    n_params = apply_lora(model, rank=rank, scale=scale)
    print(f"trainable params: {n_params:,}")
    opt = optim.Adam(learning_rate=lr)
    loss_and_grad = nn.value_and_grad(model, slot_loss)
    data = make_dataset(n_train, seed)
    random.Random(seed).shuffle(data)
    t0, running = time.perf_counter(), []
    for step, (request, y, _) in enumerate(data, start=1):
        prompt, _ = reader.prompt_ids(request)
        canvas = reader.layout(request)
        cache = reader._prefill(prompt, model.make_cache())  # weights move every step: no prefix cache
        ids = mx.array([canvas.token_ids], dtype=mx.int32)
        positions = mx.array([s.position for s in canvas.slots])
        ids[0, positions] = mx.random.randint(0, reader.vocab_size, (len(canvas.slots),)).astype(mx.int32)
        masks = model.diffusion_decoder_masks(ids, cache, None)
        targets = [y[s.question_id] for s in canvas.slots]
        loss, grads = loss_and_grad(model, ids, cache, masks, canvas, targets)
        opt.update(model, grads)
        mx.eval(model.trainable_parameters(), opt.state, loss)
        running.append(loss.item())
        if step % 25 == 0:
            print(
                f"step {step:4d} loss(avg25)={np.mean(running[-25:]):.4f} {time.perf_counter() - t0:.0f}s", flush=True
            )
    path = RUNS / "adapters" / "tickets-lora.safetensors"
    save_adapter(model, path, {"rank": rank, "scale": scale, "lr": lr, "n_train": n_train, "seed": seed})
    reader._prefix_cache.clear()
    return path


def _cmp(before: Metrics, after: Metrics) -> str:
    return f"base acc={before.accuracy:.3f} nll={before.nll:.3f} | lora acc={after.accuracy:.3f} nll={after.nll:.3f}"


def eval_tickets(reader: DiffusionReader) -> Dict[str, dict]:
    """Per-question raw metrics on the domain test half; compared against the base reads."""
    base = {json.loads(line)["i"]: json.loads(line) for line in (RUNS / "reads-tickets.jsonl").read_text().splitlines()}
    rows = []
    for i, (request, y, _) in enumerate(make_dataset(600, 1)):
        if i < 300:
            continue
        res = reader.read(request)
        rows.append({"i": i, "labels": y, "logits": {s.question_id: list(s.logits) for s in res.slots}})
    (RUNS / "reads-tickets-lora.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    out = {}
    for q in QUESTIONS:
        y = [r["labels"][q] for r in rows]
        before = evaluate([base[r["i"]]["logits"][q] for r in rows], y)
        after = evaluate([r["logits"][q] for r in rows], y)
        print(f"{q:12s} {_cmp(before, after)}")
        out[q] = {"base": before.__dict__, "lora": after.__dict__}
    return out


def eval_public(reader: DiffusionReader, per_task: int) -> Dict[str, dict]:
    """Regression check on the public benchmark items already read by the base model."""
    base = {json.loads(line)["uid"]: json.loads(line) for line in (RUNS / "reads.jsonl").read_text().splitlines()}
    seen: Dict[str, int] = {}
    out: Dict[str, List[tuple]] = {}
    for task, uid, request, target in tasks(200, 0):
        if seen.get(task, 0) >= per_task or uid not in base:
            continue
        seen[task] = seen.get(task, 0) + 1
        res = reader.read(request)
        out.setdefault(task, []).append((base[uid]["logits"], list(res.slots[0].logits), target))
    summary = {}
    for task, rs in out.items():
        y = [r[2] for r in rs]
        before, after = evaluate([r[0] for r in rs], y), evaluate([r[1] for r in rs], y)
        print(f"{task:8s} {_cmp(before, after)}")
        summary[task] = {"base": before.__dict__, "lora": after.__dict__}
    return summary


def main() -> None:
    """CLI."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-train", type=int, default=400)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--scale", type=float, default=20.0)
    parser.add_argument("--public-per-task", type=int, default=100)
    args = parser.parse_args()
    reader = DiffusionReader()
    path = train(reader, args.n_train, args.seed, args.lr, args.rank, args.scale)
    print("adapter:", path)
    report = {"tickets": eval_tickets(reader), "public": eval_public(reader, args.public_per_task), "args": vars(args)}
    (RUNS / "report-lora.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
