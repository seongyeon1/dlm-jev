"""LoRA on DiffusionGemma's attention projections, trained with a proper scoring rule on slot labels.

The encoder reuses the decoder's layers, so adapting the decoder attention adapts both.
Training loss = cross-entropy of the restricted label distribution at each answer slot,
i.e. exactly the distribution the reader serves. For single-token answers this is the
supervised equivalent of an outcome-calibration reward (docs/DESIGN.md §3).
"""

from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten
from mlx_vlm.models.diffusion_gemma.language import Router
from mlx_vlm.trainer.lora_layers import LoRALinear

from jevlike.canvas import Canvas

TARGETS = ("q_proj", "v_proj")


def _router_call(self: Router, x: mx.array) -> Tuple[mx.array, mx.array]:
    """Router.__call__ with the discrete top-k indices cut from the gradient graph.

    Same forward as mlx_vlm's; without the stop_gradient, every gather keyed by the
    indices (take_along_axis, expert sort, gather_qmm) asks for a VJP w.r.t. them once
    the hidden states depend on trainable LoRA weights.
    """
    x = mx.fast.rms_norm(x, None, self.eps)
    x = x * self.scale * self._root_size
    scores = self.proj(x)
    top_k = self.config.top_k_experts
    indices = mx.stop_gradient(mx.argpartition(scores, kth=-top_k, axis=-1)[..., -top_k:])
    weights = mx.take_along_axis(scores, indices, axis=-1)
    weights = mx.softmax(weights, axis=-1, precise=True)
    weights = weights * self.per_expert_scale[indices]
    return indices, weights


def apply_lora(model: nn.Module, rank: int = 8, scale: float = 20.0, targets: Sequence[str] = TARGETS) -> int:
    """Freeze the model and wrap the target projections of every decoder layer; returns trainable params."""
    model.freeze()
    Router.__call__ = _router_call  # type: ignore[method-assign]
    for layer in model.model.decoder.layers:
        attn = layer.self_attn
        for name in targets:
            base = getattr(attn, name, None)
            if base is not None:
                setattr(attn, name, LoRALinear.from_base(base, r=rank, scale=scale))
    # Training mode makes the MoE experts stop gradients through the router's discrete top-k indices.
    model.train()
    return sum(int(v.size) for _, v in tree_flatten(model.trainable_parameters()))  # type: ignore[union-attr,misc,str-unpack]


def save_adapter(model: nn.Module, path: Path, meta: Dict[str, Any]) -> None:
    """Write only the LoRA weights (plus metadata)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    weights = dict(tree_flatten(model.trainable_parameters()))  # type: ignore[arg-type]
    mx.save_safetensors(str(path), weights, metadata={k: str(v) for k, v in meta.items()})


def load_adapter(model: nn.Module, path: Path) -> None:
    """Wrap the model the way training did, then load the adapter weights."""
    loaded = mx.load(str(path), return_metadata=True)
    meta: Dict[str, str] = loaded[1] if isinstance(loaded, tuple) else {}
    apply_lora(model, rank=int(meta.get("rank", 8)), scale=float(meta.get("scale", 20.0)))
    model.load_weights(str(path), strict=False)
    model.eval()


def slot_loss(
    model: nn.Module, canvas_ids: mx.array, cache: List[Any], masks: Any, canvas: Canvas, targets: List[int]
) -> mx.array:
    """Mean cross-entropy of the restricted label distributions at the answer slots."""
    decoder = model.model.decoder
    positions = mx.array([s.position for s in canvas.slots])
    hidden = decoder(canvas_ids, cache=cache, decoder_attention_mask=masks)
    logits = model._softcap(decoder.embed_tokens.as_linear(hidden[:, positions])).astype(mx.float32)
    losses = []
    for i, (slot, y) in enumerate(zip(canvas.slots, targets, strict=True)):
        label = logits[0, i][mx.array(slot.token_ids)]
        losses.append(mx.logsumexp(label) - label[y])
    return mx.mean(mx.stack(losses))
