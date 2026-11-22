"""Compare pause shape fitting with direct regression and known price paths."""

from importlib import import_module
from pathlib import Path

import pytest


def test_center_slope_constraint_matches_least_squares(monkeypatch: pytest.MonkeyPatch) -> None:
    np = pytest.importorskip("numpy")
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    shape = import_module("confirm_pause_precision").candidate_module("pause_shape_confirmation")
    rng = np.random.default_rng(7291)
    x = np.array([1, 3, 4, 7, 10, 12, 13, 18, 24, 27, 30, 31, 35, 37], dtype=float)
    y = rng.normal(size=len(x))
    y -= y.mean()
    first = np.array([[7], [10], [13]])
    last = np.array([[18], [24], [27]])
    bases = (np.minimum(x - first, 0), np.clip(x - first, 0, last - first), np.maximum(x - last, 0))
    for basis in bases:
        basis -= basis.mean(axis=1, keepdims=True)
    for fraction in (0.0, 1 / 5):
        direct = []
        for left, middle, right in zip(*bases, strict=True):
            design = np.column_stack((left + fraction / 2 * middle, right + fraction / 2 * middle))
            beta = np.linalg.lstsq(design, y, rcond=None)[0]
            direct.append(np.sum((design @ beta) ** 2))
        assert shape._best_fit(*bases, y, fraction) == pytest.approx(max(direct), rel=1e-11)


def test_shape_check_distinguishes_pause_from_shallow_ramp(monkeypatch: pytest.MonkeyPatch) -> None:
    np = pytest.importorskip("numpy")
    from test_segmentation import experiment
    from test_trend_pauses import path, scan

    from regimex.directions import Direction

    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    shape = import_module("confirm_pause_precision").candidate_module("pause_shape_confirmation")
    for sign in (1, -1):
        direction = Direction.GROWTH if sign == 1 else Direction.DECLINE
        pause = scan(experiment(path(256, [(96, 144)], sign=sign)).observations, direction)
        confirmed = shape.prepare(pause, ("shape-5",)).select("shape-5")
        assert [(hit.start, hit.stop, hit.direction) for hit in confirmed] == [(96, 144, direction)]
        steps = np.arange(256)
        # Both the slowdown and the pullback pass the absolute drift proposal filter.
        for fraction in (0.16, -0.2):
            progress = steps - (1 - fraction) * np.clip(steps - 96, 0, 31)
            prices = (100 * np.exp(sign * 0.0002 * progress)).tolist()
            ramp = scan(experiment(prices).observations, direction)
            assert ramp.select(gain_multiplier=4, absolute_multiplier=1)
            assert not shape.confirm_pauses(ramp, "shape-5")
