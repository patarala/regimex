"""Check the offline pause rule's grid, semantic and provenance constraints."""

import importlib
import math
from dataclasses import replace
from pathlib import Path
from types import ModuleType

import pytest
from test_segmentation import experiment

from regimex.directions import AnalysisConfig, Direction
from regimex.segmentation import annotate


@pytest.fixture
def candidate(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    return importlib.import_module("refine_trend_pauses")


def ramp_with_pause(sign: int, scale: float) -> list[float]:
    return [
        scale * math.exp(sign * 0.00016 * (min(step, 72) + max(0, step - 103)))
        for step in range(192)
    ]


@pytest.mark.parametrize(("sign", "direction"), [(-1, Direction.DECLINE), (1, Direction.GROWTH)])
@pytest.mark.parametrize("scale", [0.01, 100.0, 1e8])
def test_observed_pause_inside_trend(
    candidate: ModuleType, sign: int, direction: Direction, scale: float
) -> None:
    grid = experiment(ramp_with_pause(sign, scale))
    config = AnalysisConfig()
    baseline = annotate(grid, config)
    assert [i.direction for i in baseline.intervals] == [direction]
    result = candidate.annotate_pauses(grid, config)
    assert result.experiment_id == grid.experiment_id
    assert result.policy != baseline.policy
    assert [(i.start, i.stop, i.direction) for i in result.intervals] == [
        (0, 72, direction),
        (72, 104, Direction.SIDEWAYS),
        (104, 192, direction),
    ]


@pytest.mark.parametrize("value", ["slow", "reversal", "shift", "oscillation"])
def test_smooth_trends_and_jumps_preserve_baseline(candidate: ModuleType, value: str) -> None:
    x = range(192)
    prices = {
        "slow": [100 * math.exp(0.00011 * i) for i in x],
        "reversal": [100 * math.exp(0.00016 * min(i, 191 - i)) for i in x],
        "shift": [100 * math.exp(0.015 * (i >= 96)) for i in x],
        "oscillation": [100 * math.exp(0.003 * math.sin(i / 5)) for i in x],
    }
    grid = experiment(prices[value])
    config = AnalysisConfig()
    assert candidate.annotate_pauses(grid, config).intervals == annotate(grid, config).intervals


@pytest.mark.parametrize("unavailable", ["insufficient_data", "no_trades"])
def test_unavailable_runs_stay_on_their_instrument(candidate: ModuleType, unavailable: str) -> None:
    first = experiment(ramp_with_pause(1, 100.0))
    prices: list[float | None] = [*ramp_with_pause(-1, 100.0)]
    counts: list[int | None] = [1] * len(prices)
    if unavailable == "no_trades":
        counts[72:104] = [0] * 32
    else:
        prices[72:104] = [None] * 32
        counts[72:104] = [None] * 32
    second = experiment(prices, counts)
    peer = second.instruments[0].model_copy(update={"instrument_id": "B/Q", "base_asset": "B"})
    rows = tuple(row.model_copy(update={"instrument_id": "B/Q"}) for row in second.observations)
    grid = replace(
        first, instruments=(*first.instruments, peer), observations=(*first.observations, *rows)
    )
    refined = candidate.annotate_pauses(grid, AnalysisConfig())
    assert [(i.start, i.stop, i.status) for i in refined.intervals if i.instrument_id == "B/Q"] == [
        (0, 72, "observed"),
        (72, 104, unavailable),
        (104, 192, "observed"),
    ]
    assert [i.direction for i in refined.intervals if i.instrument_id == "A/Q"] == [
        Direction.GROWTH,
        Direction.SIDEWAYS,
        Direction.GROWTH,
    ]
