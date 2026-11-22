"""Numerical and interval contracts for the correlation-aware price pause check."""

from importlib import import_module
from importlib.util import find_spec
from pathlib import Path
from types import ModuleType

import pytest


@pytest.fixture
def robust(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    pytest.importorskip("numpy")
    if find_spec("regimex.robust_pause_fit") is not None:
        return import_module("regimex.robust_pause_fit")
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    return import_module("robust_pause_fit")


def test_irregular_execution_innovations_match_correlated_least_squares(robust: ModuleType) -> None:
    import numpy as np

    steps = np.array([1, 2, 3, 6, 14, 15, 25], dtype=float)
    residuals = np.array([[0, 1, 2, 3, 4, -1, 2], [1, 2, -2, -3, 0, -2, 1]], dtype=float)
    covariance = 0.55 ** np.abs(steps[:, None] - steps)
    innovations = robust._innovations(residuals, 0.55 ** np.diff(steps))
    np.testing.assert_allclose(
        innovations @ innovations.T, residuals @ np.linalg.solve(covariance, residuals.T)
    )


@pytest.mark.parametrize("sign", [1, -1])
def test_close_pauses_at_opposite_parent_edges_survive_an_outlier(
    robust: ModuleType, sign: int
) -> None:
    from test_segmentation import experiment
    from test_trend_pauses import path, scan

    from regimex.directions import Direction

    direction = Direction.GROWTH if sign == 1 else Direction.DECLINE
    intervals = [(32, 72), (104, 144)]
    prices = path(176, intervals, sign=sign)
    prices[51] *= 1.005
    fitted = scan(experiment(prices).observations, direction)
    for setting in robust.SETTINGS:
        pauses = robust.detect(fitted, setting)
        assert [(p.start, p.stop) for p in pauses] == intervals
        assert all(p.direction == direction for p in pauses)


@pytest.mark.parametrize("sign", [1, -1])
def test_brief_pullback_rejects_an_otherwise_eligible_window(robust: ModuleType, sign: int) -> None:
    from test_segmentation import experiment
    from test_trend_pauses import path, scan

    from regimex.directions import Direction

    direction = Direction.GROWTH if sign == 1 else Direction.DECLINE
    prices = path(192, [(64, 96)], sign=sign, slope_fraction=-0.15)
    fitted = scan(experiment(prices).observations, direction)
    assert fitted.select(gain_multiplier=3, absolute_multiplier=1.5)
    assert all(not robust.detect(fitted, setting) for setting in robust.SETTINGS)
