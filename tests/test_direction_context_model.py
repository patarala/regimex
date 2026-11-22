"""Exercise local movement evidence, training isolation and portable inference."""

from importlib import import_module
from pathlib import Path
from types import ModuleType

import pytest


@pytest.fixture
def candidate(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    pytest.importorskip("numpy")
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    return import_module("direction_context_model")


def test_movement_with_and_without_an_intercept_change(candidate: ModuleType) -> None:
    import numpy as np
    from numpy.typing import ArrayLike
    from test_segmentation import experiment

    from regimex.directions import AnalysisConfig

    config = AnalysisConfig()
    step = np.arange(128)
    level = 0.08 * (step >= 64)
    ordinary = candidate.FEATURE_NAMES.index("1_center_movement")
    adjusted = candidate.FEATURE_NAMES.index("1_center_within_level_movement")

    def center(log_prices: ArrayLike) -> tuple[float, float]:
        observations = experiment((100 * np.exp(log_prices)).tolist()).observations
        row = candidate.features(observations, config)[64]
        return row[ordinary], row[adjusted]

    ordinary_step, adjusted_step = center(level)
    assert ordinary_step > 1
    assert adjusted_step == pytest.approx(0, abs=1e-10)
    ordinary_slope, adjusted_slope = center(0.0008 * step)
    assert ordinary_slope == pytest.approx(adjusted_slope)
    assert center(0.0008 * step + level)[1] == pytest.approx(adjusted_slope)
    assert center(-0.0008 * step + level)[1] == pytest.approx(-adjusted_slope)


def test_availability_and_local_price_units(candidate: ModuleType) -> None:
    import numpy as np
    from test_segmentation import experiment

    from regimex.directions import AnalysisConfig

    config = AnalysisConfig()
    prices = np.exp(np.random.default_rng(532).normal(0, 0.002, 96).cumsum()).tolist()
    rows = experiment(prices + [None] * 35 + prices[::-1]).observations
    full = candidate.features(rows, config)
    assert full.shape == (len(rows), len(candidate.FEATURE_NAMES))
    assert np.all(full[96:131] == 0)
    np.testing.assert_array_equal(full[:96], candidate.features(rows[:96], config))
    scaled = experiment([v * 1000 for v in prices]).observations
    np.testing.assert_allclose(full[:96], candidate.features(scaled, config), atol=1e-8)
    prices[40] = None
    short_gap = candidate.features(experiment(prices).observations, config)
    assert np.isfinite(short_gap).all()
    assert short_gap[40, candidate.FEATURE_NAMES.index("1_center_coverage")] == 31 / 32


def test_fit_respects_unknown_labels_reflection_and_round_trip(
    candidate: ModuleType, tmp_path: Path
) -> None:
    import numpy as np

    pytest.importorskip("sklearn")
    rng = np.random.default_rng(102)
    x = rng.normal(size=(90, len(candidate.FEATURE_NAMES)))
    y = np.tile([0, 1, 2], 30)
    model = candidate.fit(x, y)
    extra = np.vstack([x, np.full(x.shape[1], 1e5)])
    masked = candidate.fit(extra, np.append(y, -1))
    np.testing.assert_array_equal(masked.mean, model.mean)
    np.testing.assert_array_equal(masked.predict(x), model.predict(x))
    bundle = tmp_path / "direction.npz"
    model.save(bundle)
    probabilities = candidate.DirectionContextModel.load(bundle).predict(x)
    np.testing.assert_array_equal(probabilities, model.predict(x))
    np.testing.assert_allclose(probabilities.sum(axis=1), 1)
    reflected = x.copy()
    reflected[:, candidate._SIGN_COLUMNS] *= -1
    np.testing.assert_allclose(model.predict(reflected), probabilities[:, [0, 2, 1]], atol=1e-14)
    with np.load(bundle, allow_pickle=False) as archive:
        invalid = {key: archive[key] for key in archive.files}
    invalid["classes"] = invalid["classes"][::-1]
    np.savez(bundle, **invalid)
    with pytest.raises(ValueError, match="classes"):
        candidate.DirectionContextModel.load(bundle)
