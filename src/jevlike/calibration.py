"""Per-question-type temperature scaling fitted on outcomes, plus calibration metrics.

A slot read yields label logits. Dividing them by one scalar T per question type
keeps the argmax (accuracy unchanged) and only reshapes the probabilities; T is
chosen to minimise negative log-likelihood of the true labels.
"""

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
from numpy.typing import ArrayLike

_LOG_T_MIN, _LOG_T_MAX = math.log(0.05), math.log(20.0)


def softmax(logits: Sequence[float], temperature: float = 1.0) -> np.ndarray:
    """Numerically stable softmax of ``logits / temperature``."""
    z = np.asarray(logits, dtype=np.float64) / temperature
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


def normalized_confidence(probs: ArrayLike) -> float:
    """``1 - H(p) / ln K``: 0 for a uniform distribution, 1 for a one-hot one."""
    p = np.asarray(probs, dtype=np.float64)
    if p.size < 2:
        return 1.0
    nz = p[p > 0]
    entropy = float(-(nz * np.log(nz)).sum())
    return max(0.0, 1.0 - entropy / math.log(p.size))


def nll(logits: Sequence[Sequence[float]], targets: Sequence[int], temperature: float = 1.0) -> float:
    """Mean negative log-likelihood of ``targets``."""
    total = 0.0
    for row, y in zip(logits, targets, strict=True):
        p = softmax(row, temperature)[y]
        total -= math.log(max(float(p), 1e-12))
    return total / max(len(targets), 1)


def fit_temperature(logits: Sequence[Sequence[float]], targets: Sequence[int]) -> float:
    """Golden-section search over log T for the NLL-minimising temperature."""
    if not targets:
        return 1.0
    ratio = (math.sqrt(5) - 1) / 2
    lo, hi = _LOG_T_MIN, _LOG_T_MAX
    a, b = hi - ratio * (hi - lo), lo + ratio * (hi - lo)
    fa, fb = nll(logits, targets, math.exp(a)), nll(logits, targets, math.exp(b))
    for _ in range(60):
        if fa < fb:
            hi, b, fb = b, a, fa
            a = hi - ratio * (hi - lo)
            fa = nll(logits, targets, math.exp(a))
        else:
            lo, a, fa = a, b, fb
            b = lo + ratio * (hi - lo)
            fb = nll(logits, targets, math.exp(b))
    return math.exp((lo + hi) / 2)


@dataclass(frozen=True)
class Metrics:
    """Accuracy and calibration quality of a set of predictions."""

    n: int
    accuracy: float
    nll: float
    brier: float
    ece: float


def evaluate(
    logits: Sequence[Sequence[float]], targets: Sequence[int], temperature: float = 1.0, bins: int = 10
) -> Metrics:
    """Compute accuracy, NLL, multi-class Brier and top-label ECE."""
    n = len(targets)
    if n == 0:
        return Metrics(0, 0.0, 0.0, 0.0, 0.0)
    probs = [softmax(row, temperature) for row in logits]
    top = np.array([float(p.max()) for p in probs])
    correct = np.array([int(p.argmax()) == y for p, y in zip(probs, targets, strict=True)], dtype=np.float64)
    brier = float(np.mean([((p - np.eye(p.size)[y]) ** 2).sum() for p, y in zip(probs, targets, strict=True)]))
    ece = 0.0
    edges = np.linspace(0.0, 1.0, bins + 1)
    for i in range(bins):
        in_bin = (top > edges[i]) & (top <= edges[i + 1]) if i else (top <= edges[1])
        if in_bin.any():
            ece += in_bin.mean() * abs(correct[in_bin].mean() - top[in_bin].mean())
    return Metrics(n, float(correct.mean()), nll(logits, targets, temperature), brier, float(ece))


@dataclass
class Calibrator:
    """Temperature per question type (``noul`` / ``choice`` / ``score``); missing types use T=1."""

    temperatures: Dict[str, float] = field(default_factory=dict)
    name: str = "uncalibrated"

    def temperature(self, question_type: str) -> float:
        """Temperature for a question type."""
        return self.temperatures.get(question_type, 1.0)

    def save(self, path: Path) -> None:
        """Write as JSON."""
        path.write_text(json.dumps({"name": self.name, "temperatures": self.temperatures}, indent=2))

    @classmethod
    def load(cls, path: Path) -> "Calibrator":
        """Read from JSON written by :meth:`save`."""
        data = json.loads(path.read_text())
        return cls(temperatures={k: float(v) for k, v in data["temperatures"].items()}, name=data["name"])

    @classmethod
    def fit(cls, samples: Dict[str, List[tuple]], name: str) -> "Calibrator":
        """Fit from ``{type: [(label_logits, target_index), ...]}``."""
        temps = {}
        for question_type, rows in samples.items():
            temps[question_type] = fit_temperature([r[0] for r in rows], [r[1] for r in rows])
        return cls(temperatures=temps, name=name)
