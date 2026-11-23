from importlib import import_module
from pathlib import Path

import pytest


def test_local_forest_keeps_piece_and_short_window_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    np = pytest.importorskip("numpy")
    from test_segmentation import experiment

    from regimex.directions import AnalysisConfig

    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    candidate = import_module("local_adaptive_forest")
    steps = np.arange(128)
    prices = np.exp(0.0005 * (np.minimum(steps, 44) + np.maximum(0, steps - 85)))
    rows = experiment(prices.tolist() + [None] * 40 + prices[::-1].tolist()).observations
    config = AnalysisConfig()
    scores = candidate.features(rows, config)
    assert scores.shape == (len(rows), 30)
    assert len(set(candidate.FEATURE_NAMES)) == scores.shape[1]
    np.testing.assert_array_equal(scores[:128], candidate.features(rows[:128], config))
    assert not np.any(scores[128:168])
    slope = candidate.FEATURE_NAMES.index("1_center_within_level_movement")
    assert abs(scores[64, slope]) < 1e-6
    assert scores[20, slope] > 0.5
    assert scores[168 + 20, slope] < -0.5
