from importlib import import_module
from pathlib import Path
from types import ModuleType

import pytest


@pytest.fixture
def adaptive() -> ModuleType:
    pytest.importorskip("sklearn")
    return import_module("regimex.backends.adaptive_forest")


def test_adaptive_context_obeys_price_units_and_unavailable_blocks(adaptive: ModuleType) -> None:
    import numpy as np
    from test_segmentation import experiment

    from regimex.directions import AnalysisConfig

    config = AnalysisConfig()
    prices = (100 * np.exp(0.0003 * np.arange(96))).tolist()
    rows = experiment(prices + [None] * 40 + list(reversed(prices))).observations
    result = adaptive.features(rows, config)
    assert np.isfinite(result).all()
    assert np.all(result[96:136] == 0)
    np.testing.assert_allclose(result[:96], adaptive.features(rows[:96], config))
    changed = experiment([100.0] * 96 + [None] * 40 + list(reversed(prices))).observations
    np.testing.assert_array_equal(result[136:], adaptive.features(changed, config)[136:])
    scaled = tuple(
        r.model_copy(update={"last_price": 2 * r.last_price if r.last_price else None})
        for r in rows
    )
    np.testing.assert_allclose(result, adaptive.features(scaled, config), atol=1e-10)


def test_isolated_level_change_has_no_adjusted_slope(adaptive: ModuleType) -> None:
    import numpy as np

    values = np.repeat([0, 0.01], 32)
    move, _, adjusted, _, _, _ = adaptive._describe(values, 0.005)
    assert move > 1 and adjusted == pytest.approx(0, abs=1e-12)


@pytest.mark.parametrize("noise", [1e154, 1e155])
def test_extreme_valid_scales_keep_finite_signed_features(
    adaptive: ModuleType, noise: float
) -> None:
    import math

    import numpy as np
    from test_segmentation import experiment

    from regimex.directions import AnalysisConfig

    scale = 1e-312
    profile = AnalysisConfig(minimum_relative_change=scale, noise_floor=noise)
    for movement in (0.5, -0.5):
        rows = experiment(np.exp(np.linspace(0, movement, 96)).tolist()).observations
        features = adaptive.features(rows, profile)
        assert np.isfinite(features).all()
        # With no splits the first column is the fitted movement over the whole block.
        expected = math.copysign(math.log(2 * abs(movement)) - math.log(scale), movement)
        np.testing.assert_allclose(features[:, 0], expected, rtol=1e-14)


def test_numeric_forest_matches_fitted_trees_and_rejects_cycles(
    adaptive: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import numpy as np
    from sklearn.ensemble import ExtraTreesClassifier  # type: ignore[import-untyped]

    from regimex.resource_limits import ResourceBudgetExceeded

    estimator = ExtraTreesClassifier(n_estimators=3, max_depth=4, random_state=173)
    monkeypatch.setattr("sklearn.ensemble.ExtraTreesClassifier", lambda **kwargs: estimator)
    rng = np.random.default_rng(422)
    x = rng.normal(size=(120, adaptive._WIDTH))
    targets = np.digitize(x[:, 0], (-0.5, 0.4))
    targets[::7] = -1
    model = adaptive.fit(x, targets)
    bundle = tmp_path / "trees.npz"
    model.save(bundle)
    restored = adaptive.DirectionContextModel.load(bundle, trees=3, max_depth=4)
    expected = estimator.predict_proba(x)
    np.testing.assert_allclose(restored.predict(x), expected, atol=1e-14)
    np.testing.assert_allclose(restored.predict(x[:9]), expected[:9], atol=1e-14)
    with pytest.raises(ResourceBudgetExceeded):
        adaptive.DirectionContextModel.load(bundle, max_bytes=restored.numeric_bytes - 1)
    with pytest.raises(ValueError, match="depth"):
        adaptive.DirectionContextModel.load(bundle, max_depth=1, trees=None)
    with pytest.raises(ValueError, match="read-only"):
        restored.children[0, 0] = 0
    corrupted = tmp_path / "cycle.npz"
    with np.load(bundle, allow_pickle=False) as archive:
        members = dict(archive)
    members["children"][0, 0] = 0
    np.savez(corrupted, **members)
    with pytest.raises(ValueError, match="children"):
        adaptive.DirectionContextModel.load(corrupted)


def test_forest_restores_repeated_feature_runs_to_the_original_grid(adaptive: ModuleType) -> None:
    import numpy as np

    rng = np.random.default_rng(31)
    x = rng.normal(size=(120, len(adaptive.FEATURE_NAMES)))
    model = adaptive.fit(x, np.digitize(x[:, 0], (-0.5, 0.5)), trees=5, max_depth=5)
    expected = model.predict(x)
    order = [7, 7, 2, 3, 3, 3, 7, 7, 8]
    repeated = x[order]
    # The estimator sees the same float32 row on both sides of this float64 change.
    repeated[1] = np.nextafter(repeated[1], np.inf)
    np.testing.assert_array_equal(repeated[0].astype("float32"), repeated[1].astype("float32"))
    np.testing.assert_array_equal(model.predict(repeated), expected[order])
    np.testing.assert_array_equal(model.predict(np.repeat(x[:1], 256, axis=0)), expected[[0] * 256])
    assert model.predict(x[:0]).shape == (0, 3)
    dense = rng.normal(size=(1025, x.shape[1]))
    np.testing.assert_array_equal(
        model.predict(dense),
        np.concatenate([model.predict(chunk) for chunk in np.array_split(dense, 3)]),
    )
