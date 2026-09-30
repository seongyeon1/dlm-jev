"""Calibration tests."""

import numpy as np

from dlm_jev.calibration import Calibrator, evaluate, fit_temperature, normalized_confidence, softmax


def test_confidence_bounds() -> None:
    """Uniform → 0, one-hot → 1."""
    assert normalized_confidence([0.25] * 4) == 0.0
    assert normalized_confidence([1.0, 0.0, 0.0]) == 1.0


def test_fit_recovers_overconfidence() -> None:
    """Logits sharpened 3x from a calibrated source should fit T≈3 and lower NLL."""
    # Given: labels drawn from softmax(z), then logits reported as 3*z
    rng = np.random.default_rng(0)
    z = rng.normal(size=(4000, 3))
    y = [int(rng.choice(3, p=softmax(row))) for row in z]
    overconfident = (3 * z).tolist()
    # When
    t = fit_temperature(overconfident, y)
    # Then
    assert 2.6 < t < 3.4
    assert evaluate(overconfident, y, t).nll < evaluate(overconfident, y).nll
    assert evaluate(overconfident, y, t).accuracy == evaluate(overconfident, y).accuracy


def test_calibrator_roundtrip(tmp_path) -> None:
    """save/load keeps temperatures; unknown types default to 1."""
    c = Calibrator.fit({"noul": [([2.0, -2.0], 0), ([1.0, 0.0], 1)]}, name="t")
    c.save(tmp_path / "c.json")
    loaded = Calibrator.load(tmp_path / "c.json")
    assert loaded.temperature("noul") == c.temperature("noul")
    assert loaded.temperature("score") == 1.0
