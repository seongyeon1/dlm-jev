# jevlike

A [Jev](https://en.wikipedia.org/wiki/Jev_(AI_model))-style typed decision engine built on a diffusion LM (DiffusionGemma 26B-A4B, MLX 4-bit).
It doesn't generate text. It returns **a probability distribution and a confidence** for each typed question (`noul` / `choice` / `score`) about a given `state`.
All questions are read in a single decoder pass.

Design and rationale: [docs/DESIGN.md](docs/DESIGN.md)

## Quick start (Apple silicon, 32GB+ memory)

```bash
uv sync
uv run python scripts/smoke.py                 # first run downloads the model (~15GB)
uv run python scripts/benchmark.py --per-task 200   # SST-2 / AG News / Yelp → fits runs/calibration.json
JEVLIKE_CALIBRATION=runs/calibration.json uv run jevlike-serve   # 127.0.0.1:8765
# domain adapter: JEVLIKE_ADAPTER=runs/adapters/tickets-lora.safetensors (after scripts/train_lora.py)
```

```bash
curl -s localhost:8765/v1/decide -H 'content-type: application/json' -d '{
  "state": "Verify your password within 24h at http://paypa1-secure.com",
  "questions": {
    "phishing": {"type": "noul", "instructions": "This email is a phishing attempt."},
    "route": {"type": "choice", "instructions": "Queue?", "options": ["inbox", "security review", "spam"]},
    "urgency": {"type": "score", "instructions": "Urgency?", "levels": ["none", "low", "high"]}
  }
}'
```

## Layout

| Path | Role |
|---|---|
| `src/jevlike/schema.py` | Request/response types |
| `src/jevlike/canvas.py` | Prompt and canvas assembly, slot positions, label tokens (single source) |
| `src/jevlike/engine.py` | Question-prefix cache → state prefill → batched slot read → typed answer |
| `src/jevlike/calibration.py` | Per-type temperature scaling, NLL / Brier / ECE |
| `src/jevlike/server.py` | FastAPI `POST /v1/decide` |
| `scripts/benchmark.py` | Accuracy and calibration measured on public datasets |
| `scripts/latency.py`, `scripts/reread_value.py` | Latency profile; whether re-reads pay for themselves |
| `src/jevlike/synth.py`, `scripts/domain_eval.py` | Synthetic support tickets (ground truth by construction), per-question calibration |
| `src/jevlike/lora.py`, `scripts/train_lora.py`, `scripts/shift_eval.py` | L2 LoRA on slot-label CE; does it learn the policy or the templates |
| `scripts/interference.py`, `scripts/ar_baseline.py` | Multi-question interference; autoregressive baseline |

## Tests

```bash
uv run pytest -q          # no model needed (stub tokenizer / stub reader)
uv run ruff check . && uv run mypy src --ignore-missing-imports
```
