"""Synthetic ticket ground-truth rules."""

import random

from dlm_jev.synth import CATEGORIES, QUESTIONS, Ticket, labels, make_dataset, render


def _t(cat: str, **kw: object) -> Ticket:
    base = dict(
        users_affected=1,
        workaround=False,
        admin=False,
        wrong_charge=False,
        churn=False,
        tone="calm",
        decoy_cancel=False,
        decoy_billing_words=False,
    )
    base.update(kw)
    return Ticket(category=CATEGORIES.index(cat), **base)  # type: ignore[arg-type]


def test_urgency_follows_impact_not_tone() -> None:
    """Angry how-to is still low; a company-wide outage is critical."""
    assert labels(_t("사용 방법", tone="angry"))["urgency"] == 0
    assert labels(_t("서비스 장애", users_affected=200))["urgency"] == 3
    assert labels(_t("버그", workaround=True, users_affected=200))["urgency"] == 1


def test_decoy_cancel_is_not_churn() -> None:
    """'취소' in a non-churn sense appears in the text but the label stays no."""
    t = _t("버그", decoy_cancel=True)
    text = render(t, random.Random(0))
    assert "취소" in text
    assert labels(t)["churn"] == 1


def test_dataset_is_deterministic_and_covers_all_categories() -> None:
    """Same seed, same data; every category appears."""
    a, b = make_dataset(200, seed=3), make_dataset(200, seed=3)
    assert [r.state for r, _, _ in a] == [r.state for r, _, _ in b]
    assert {y["category"] for _, y, _ in a} == set(range(len(CATEGORIES)))
    assert set(a[0][0].questions) == set(QUESTIONS)


def test_per_question_temperature_wins_over_type() -> None:
    """A q:<id> fit overrides the per-type temperature; other ids fall back to the type."""
    from dlm_jev.calibration import Calibrator

    c = Calibrator(temperatures={"noul": 2.0, "q:churn": 5.0})
    assert c.temperature("noul", "churn") == 5.0
    assert c.temperature("noul", "engineering") == 2.0
