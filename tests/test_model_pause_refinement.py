"""The second pass only examines trends that the model actually predicts."""

import importlib
from pathlib import Path

import pytest
from test_segmentation import experiment
from test_trend_pause_refinement import ramp_with_pause

from regimex.directions import AnalysisConfig, Direction
from regimex.segmentation import _interval


def test_refinement_uses_decoded_direction_and_original_instrument_grid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    refine = importlib.import_module("refine_model_intervals").refine_decoded
    rows = experiment(ramp_with_pause(1, 100.0)).observations
    config = AnalysisConfig()
    growth = (_interval(rows, 0, len(rows), Direction.GROWTH, "observed"),)
    sideways = (_interval(rows, 0, len(rows), Direction.SIDEWAYS, "observed"),)
    result = refine(rows, growth, config)
    assert [(i.start, i.stop, i.direction) for i in result] == [
        (0, 72, Direction.GROWTH),
        (72, 104, Direction.SIDEWAYS),
        (104, 192, Direction.GROWTH),
    ]
    assert refine(rows, sideways, config) == sideways
    with pytest.raises(ValueError, match="original grid"):
        refine(rows[::-1], growth, config)
