"""Confirm near-level prices against movement continuing in either direction."""

from importlib import import_module
from pathlib import Path

import pytest


@pytest.mark.parametrize("central_fraction", [-0.18, 0.0, 0.18])
def test_flat_prices_and_bidirectional_slow_trends(
    monkeypatch: pytest.MonkeyPatch, central_fraction: float
) -> None:
    np = pytest.importorskip("numpy")
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    matcher = import_module("pause_flank_confirmation")
    from test_segmentation import experiment
    from test_trend_pauses import path, scan

    from regimex.directions import Direction

    for sign, direction in ((1, Direction.GROWTH), (-1, Direction.DECLINE)):
        prices = path(256, [(96, 128)], sign=sign, slope_fraction=central_fraction)
        series = scan(experiment(prices).observations, direction)
        assert series.select(absolute_multiplier=2)
        confirmed = matcher.prepare(series)
        for setting in matcher.SETTINGS:
            pauses = confirmed.select(setting)
            if central_fraction == 0:
                assert [(p.start, p.stop) for p in pauses] == [(96, 128)]
                assert np.isclose(pauses[0].price_change_percent, 0)
            else:
                assert not pauses


def test_clustered_execution_times_give_no_drift_estimate(monkeypatch: pytest.MonkeyPatch) -> None:
    np = pytest.importorskip("numpy")
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    matcher = import_module("pause_flank_confirmation")
    # Each cluster satisfies the count threshold but supplies virtually one distinct time.
    times = np.repeat([-0.8, 0.4, 1.8], 5) + np.tile(np.arange(5) * 1e-11, 3)
    assert np.isinf(matcher._continuation_ratio(times, np.sin(times), 31 / 32, 4, 32))
