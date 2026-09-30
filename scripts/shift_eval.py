"""Did LoRA learn the triage policy or the training templates?

Reads tickets rendered with ``render_shifted`` (same policy and labels, none of the
training phrasing) with the base model, then with the LoRA adapter loaded onto the
same model, and compares them item by item.

    uv run python scripts/shift_eval.py --n 300
"""

import argparse
import json
import sys
from math import comb
from pathlib import Path
from typing import Dict, List

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from benchmark import RUNS  # noqa: E402

from dlm_jev.calibration import evaluate  # noqa: E402
from dlm_jev.engine import DiffusionReader  # noqa: E402
from dlm_jev.lora import load_adapter  # noqa: E402
from dlm_jev.synth import QUESTIONS, make_dataset  # noqa: E402


def mcnemar(b: int, c: int) -> float:
    """Exact two-sided McNemar p-value from the discordant counts."""
    n, k = b + c, min(b, c)
    return 1.0 if n == 0 else min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2**n)


def read_all(reader: DiffusionReader, data: list) -> List[Dict[str, List[float]]]:
    """Raw label logits per question for every ticket."""
    return [{s.question_id: list(s.logits) for s in reader.read(req).slots} for req, _, _ in data]


def main() -> None:
    """Base reads, then LoRA reads, then a paired comparison per question."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=300)
    parser.add_argument("--seed", type=int, default=21)
    parser.add_argument("--adapter", default=str(RUNS / "adapters" / "tickets-lora.safetensors"))
    args = parser.parse_args()
    data = make_dataset(args.n, args.seed, shifted=True)
    reader = DiffusionReader()
    base = read_all(reader, data)
    load_adapter(reader.model, Path(args.adapter))
    reader._prefix_cache.clear()
    lora = read_all(reader, data)
    (RUNS / "reads-tickets-shifted.jsonl").write_text(
        "\n".join(
            json.dumps({"labels": y, "base": b, "lora": a}) for (_, y, _), b, a in zip(data, base, lora, strict=True)
        )
    )
    report = {}
    for q in QUESTIONS:
        y = [d[1][q] for d in data]
        mb, ml = evaluate([b[q] for b in base], y), evaluate([a[q] for a in lora], y)
        ok_b = [int(np.argmax(b[q]) == t) for b, t in zip(base, y, strict=True)]
        ok_l = [int(np.argmax(a[q]) == t) for a, t in zip(lora, y, strict=True)]
        only_b = sum(x and not z for x, z in zip(ok_b, ok_l, strict=True))
        only_l = sum(z and not x for x, z in zip(ok_b, ok_l, strict=True))
        p = mcnemar(only_b, only_l)
        print(
            f"{q:12s} base acc={mb.accuracy:.3f} nll={mb.nll:.3f} | lora acc={ml.accuracy:.3f} nll={ml.nll:.3f} "
            f"| base-only={only_b} lora-only={only_l} p={p:.3f}"
        )
        report[q] = {"base": mb.__dict__, "lora": ml.__dict__, "base_only": only_b, "lora_only": only_l, "p": p}
    (RUNS / "report-shifted.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
