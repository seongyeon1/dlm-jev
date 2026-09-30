"""DiffusionGemma slot reader on MLX: prefill the prompt once, read every answer slot in one decoder pass."""

import copy
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import mlx.core as mx
import numpy as np
from mlx_vlm.models.diffusion_gemma.language import _cache_state
from numpy.typing import ArrayLike

from jevlike.calibration import Calibrator, normalized_confidence, softmax
from jevlike.canvas import SYSTEM_PROMPT, Canvas, build_canvas, render_user_parts
from jevlike.schema import (
    Answer,
    ChoiceAnswer,
    ChoiceQuestion,
    DecideRequest,
    DecideResponse,
    NoulAnswer,
    Question,
    ScoreAnswer,
    ScoreQuestion,
    Usage,
)

DEFAULT_MODEL = "mlx-community/diffusiongemma-26B-A4B-it-4bit"
EMPTY_THOUGHT = "<|channel>thought\n<channel|>"


class _BroadcastCache:
    """Read-only view of one encoder cache layer, broadcast to batch ``n`` for batched decoder reads."""

    def __init__(self, cache: Any, n: int) -> None:
        keys, values = _cache_state(cache)
        self.keys = keys
        self.offset = cache.offset
        self.decoder_state = (
            mx.broadcast_to(keys, (n, *keys.shape[1:])),
            mx.broadcast_to(values, (n, *values.shape[1:])),
        )


def _clone_cache(cache: Any) -> Any:
    """Copy one encoder cache layer with fresh arrays: prefill writes into its buffers in place."""
    clone = copy.copy(cache)
    if cache.keys is not None:
        clone.keys, clone.values = mx.array(cache.keys), mx.array(cache.values)
    return clone


@dataclass(frozen=True)
class SlotRead:
    """Label logits for one slot (log of the read-averaged label probabilities)."""

    question_id: str
    logits: Tuple[float, ...]


@dataclass(frozen=True)
class ReadResult:
    """Raw slot reads plus accounting for one request."""

    slots: List[SlotRead]
    timing_ms: Dict[str, float]
    prompt_tokens: int
    cached_tokens: int
    canvas_tokens: int
    reads: int


class DiffusionReader:
    """Loads DiffusionGemma and answers typed questions by reading canvas slots."""

    def __init__(
        self,
        model_path: str = DEFAULT_MODEL,
        calibrator: Optional[Calibrator] = None,
        reread_below_confidence: float = 0.9,
        extra_reads: int = 0,
        prefix_cache_size: int = 32,
        seed: int = 0,
    ) -> None:
        from mlx_vlm import load

        self.model_path = model_path
        self.model, self.processor = load(model_path)
        self.tokenizer: Any = getattr(self.processor, "tokenizer", self.processor)
        self.vocab_size = int(self.model.config.text_config.vocab_size)
        self.calibrator = calibrator or Calibrator()
        self.reread_below_confidence = reread_below_confidence
        self.extra_reads = extra_reads
        self._end_ids = self._response_end_ids()
        # The instruct model opens its reply with an (empty) thought channel; the canvas does the same.
        self._lead_ids = self.tokenizer.encode(EMPTY_THOUGHT, add_special_tokens=False)
        self._pad_id = int(self.tokenizer.pad_token_id or 0)
        self._lock = threading.Lock()
        # Encoder caches keyed by the question-set prefix; never mutated, cloned per request.
        self._prefix_cache: OrderedDict[Tuple[int, ...], List[Any]] = OrderedDict()
        self._prefix_cache_size = prefix_cache_size
        mx.random.seed(seed)

    def _chat_text(self, user: str, assistant: Optional[str] = None) -> str:
        messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]
        if assistant is not None:
            messages.append({"role": "assistant", "content": assistant})
        return str(
            self.tokenizer.apply_chat_template(messages, add_generation_prompt=assistant is None, tokenize=False)
        )

    def _encode(self, text: str) -> List[int]:
        # The chat template already carries <bos>.
        return [int(i) for i in self.tokenizer.encode(text, add_special_tokens=False)]

    def _chat(self, user: str, assistant: Optional[str] = None) -> List[int]:
        return self._encode(self._chat_text(user, assistant))

    def _response_end_ids(self) -> List[int]:
        """Tokens the chat template puts after an assistant reply (e.g. end-of-turn)."""
        marker = "ZQXJ"
        full = self._chat("x", marker)
        marker_ids = self.tokenizer.encode(marker, add_special_tokens=False)
        for i in range(len(full) - len(marker_ids), -1, -1):
            if full[i : i + len(marker_ids)] == marker_ids:
                tail = full[i + len(marker_ids) :]
                return [t for t in tail if t not in self.tokenizer.encode("\n", add_special_tokens=False)][:1]
        return [int(self.tokenizer.eos_token_id)]

    def prompt_ids(self, request: DecideRequest) -> Tuple[List[int], int]:
        """Full prompt ids and the length of the reusable question prefix (0 if it doesn't split cleanly)."""
        head, tail = render_user_parts(request.state, dict(request.questions))
        text = self._chat_text(head + tail)
        ids = self._encode(text)
        prefix = self._encode(text[: text.index(head) + len(head)])
        # Only reuse when the prefix tokenizes identically on its own (a token boundary at the split).
        return ids, len(prefix) if ids[: len(prefix)] == prefix and len(prefix) < len(ids) else 0

    def _prefill(self, ids: List[int], cache: List[Any]) -> List[Any]:
        cache = self.model.diffusion_prefill_cache(
            mx.array([ids]),
            attention_mask=None,
            cache=cache,
            pixel_values=None,
            mm_token_type_ids=None,
            prefill_step_size=None,
            chunk_prefill=False,
        )
        mx.eval([c.state for c in cache])
        return list(cache)

    def _encoder_cache(self, ids: List[int], n_prefix: int) -> Tuple[List[Any], int]:
        """Encoder cache for ``ids``, reusing a cached question prefix; returns (cache, cached_tokens)."""
        if not n_prefix:
            return self._prefill(ids, self.model.make_cache()), 0
        key = tuple(ids[:n_prefix])
        base = self._prefix_cache.get(key)
        hit = base is not None
        if base is None:
            base = self._prefill(ids[:n_prefix], self.model.make_cache())
            self._prefix_cache[key] = base
            if len(self._prefix_cache) > self._prefix_cache_size:
                self._prefix_cache.popitem(last=False)
        else:
            self._prefix_cache.move_to_end(key)
        cache = self._prefill(ids[n_prefix:], [_clone_cache(c) for c in base])
        return cache, n_prefix if hit else 0

    def layout(self, request: DecideRequest) -> Canvas:
        """Canvas for a request."""
        return build_canvas(
            self.tokenizer,
            dict(request.questions),
            noise_id=0,
            end_ids=self._end_ids,
            lead_ids=self._lead_ids,
            pad_id=self._pad_id,
        )

    def _read_batch(self, canvas: Canvas, cache: Any, n: int) -> List[List[np.ndarray]]:
        """``n`` independent reads (fresh slot noise per row) in one decoder pass.

        The encoder cache is shared read-only through a batch-``n`` view, and the
        vocab projection runs only at slot positions instead of the whole canvas.
        """
        views = [_BroadcastCache(c, n) for c in cache]
        ids = mx.broadcast_to(mx.array([canvas.token_ids], dtype=mx.int32), (n, len(canvas.token_ids)))
        positions = mx.array([s.position for s in canvas.slots])
        noise = mx.random.randint(0, self.vocab_size, (n, len(canvas.slots))).astype(mx.int32)
        ids = mx.array(ids)
        ids[:, positions] = noise
        masks = self.model.diffusion_decoder_masks(ids, views, None)
        decoder = self.model.model.decoder
        hidden = decoder(ids, cache=views, decoder_attention_mask=masks)
        logits = self.model._softcap(decoder.embed_tokens.as_linear(hidden[:, positions])).astype(mx.float32)
        per_slot = []
        for i, slot in enumerate(canvas.slots):
            label = logits[:, i][:, mx.array(slot.token_ids)]
            per_slot.append(label - mx.logsumexp(label, axis=-1, keepdims=True))
        mx.eval(per_slot)
        probs = [np.exp(np.array(p, dtype=np.float64)) for p in per_slot]  # slot -> (n, K)
        return [[probs[i][row] for i in range(len(canvas.slots))] for row in range(n)]

    def read(self, request: DecideRequest, use_prefix_cache: bool = True) -> ReadResult:
        """Raw slot reads for a request."""
        with self._lock:
            t0 = time.perf_counter()
            prompt, n_prefix = self.prompt_ids(request)
            canvas = self.layout(request)
            cache, cached = self._encoder_cache(prompt, n_prefix if use_prefix_cache else 0)
            t1 = time.perf_counter()
            reads = self._read_batch(canvas, cache, 1)
            # Off by default: on AG News / Yelp, averaging 4 reads changed 0/100 and 4/100 answers
            # for ~90% more decode time (docs/DESIGN.md §6). Calibration buys far more NLL.
            if self.extra_reads and min(normalized_confidence(p) for p in reads[0]) < self.reread_below_confidence:
                reads += self._read_batch(canvas, cache, self.extra_reads)
            t2 = time.perf_counter()
        slot_reads = []
        for i, slot in enumerate(canvas.slots):
            mean = np.mean([r[i] for r in reads], axis=0)
            slot_reads.append(SlotRead(slot.question_id, tuple(np.log(np.maximum(mean, 1e-12)).tolist())))
        timing = {"prefill": (t1 - t0) * 1e3, "decode": (t2 - t1) * 1e3, "total": (t2 - t0) * 1e3}
        return ReadResult(slot_reads, timing, len(prompt), cached, len(canvas.token_ids), len(reads))

    def decide(self, request: DecideRequest) -> DecideResponse:
        """Answer every question in the request."""
        result = self.read(request)
        answers: Dict[str, Answer] = {}
        for sr in result.slots:
            question = request.questions[sr.question_id]
            t = self.calibrator.temperature(question.type, sr.question_id) if request.calibrated else 1.0
            answers[sr.question_id] = to_answer(question, softmax(sr.logits, t))
        return DecideResponse(
            model=self.model_path,
            answers=answers,
            usage=Usage(
                input_tokens=result.prompt_tokens,
                cached_tokens=result.cached_tokens,
                canvas_tokens=result.canvas_tokens,
                reads=result.reads,
            ),
            timing_ms=result.timing_ms,
            calibration=self.calibrator.name if request.calibrated else None,
        )


def to_answer(question: Question, probs: ArrayLike) -> Answer:
    """Turn a label distribution into the typed answer for ``question``."""
    p = np.asarray(probs, dtype=np.float64)
    confidence = normalized_confidence(p)
    best = int(p.argmax())
    if isinstance(question, ChoiceQuestion):
        return ChoiceAnswer(
            choice=question.options[best],
            probabilities={o: float(v) for o, v in zip(question.options, p, strict=True)},
            confidence=confidence,
        )
    if isinstance(question, ScoreQuestion):
        return ScoreAnswer(
            score=best,
            level=question.levels[best],
            expected=float((np.arange(p.size) * p).sum()),
            probabilities={lvl: float(v) for lvl, v in zip(question.levels, p, strict=True)},
            confidence=confidence,
        )
    return NoulAnswer(noul=float(p[0]), confidence=confidence)
