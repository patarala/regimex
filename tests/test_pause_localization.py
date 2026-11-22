"""Boundary uncertainty and moving-flank contracts for the pause detector."""

from importlib import import_module
from importlib.util import find_spec
from pathlib import Path
from types import ModuleType

import pytest


@pytest.fixture
def localize(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    pytest.importorskip("numpy")
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    name = "regimex.localize_pause" if find_spec("regimex.localize_pause") else "localize_pause"
    return import_module(name)


def test_prefix_residuals_match_direct_fit_on_irregular_executions(localize: ModuleType) -> None:
    import numpy as np

    x = np.array([1, 3, 5, 10, 12, 15, 17, 22, 23, 28, 29, 32, 37])
    first, last = np.array([6, 10, 12]), np.array([22, 24, 28])
    y = np.random.default_rng(139).normal(size=len(x))
    y -= y.mean()
    residuals = localize._residuals(x, y, first, last)
    for values, fraction in zip(residuals, (0.0, 1 / 6, -1 / 6), strict=True):
        for actual, a, b in zip(values, first, last, strict=True):
            mid = np.clip(x - a, 0, b - a) * fraction / 2
            design = np.column_stack(
                (np.ones(len(x)), np.minimum(x - a, 0) + mid, np.maximum(x - b, 0) + mid)
            )
            fitted = design @ np.linalg.lstsq(design, y, rcond=None)[0]
            assert actual == pytest.approx(np.sum((y - fitted) ** 2), rel=1e-11)


@pytest.mark.parametrize("sign", [1, -1])
@pytest.mark.parametrize("fraction", [0.0, -0.2])
def test_nearby_event_does_not_distort_the_moving_flank(
    localize: ModuleType, sign: int, fraction: float
) -> None:
    from test_segmentation import experiment
    from test_trend_pauses import path, scan

    from regimex.directions import Direction

    direction = Direction.GROWTH if sign == 1 else Direction.DECLINE
    bounds = [(32, 72), (104, 144)]
    series = experiment(path(176, bounds, sign=sign, slope_fraction=fraction))
    fitted = scan(series.observations, direction)
    assert fitted.select()
    for setting in localize.SETTINGS:
        hits = localize.detect(fitted, setting)
        assert [(hit.start, hit.stop) for hit in hits] == (bounds if fraction == 0 else [])
        assert all(hit.direction == direction for hit in hits)
