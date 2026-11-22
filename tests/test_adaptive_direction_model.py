from importlib import import_module
from pathlib import Path
from types import ModuleType

import pytest


@pytest.fixture
def adaptive(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    pytest.importorskip("sklearn")
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    return import_module("adaptive_direction_model")


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


def test_numeric_forest_matches_fitted_trees_and_rejects_cycles(
    adaptive: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import numpy as np
    from sklearn.ensemble import ExtraTreesClassifier  # type: ignore[import-untyped]

    estimator = ExtraTreesClassifier(n_estimators=3, max_depth=4, random_state=173)
    monkeypatch.setattr(adaptive, "ExtraTreesClassifier", lambda **kwargs: estimator)
    rng = np.random.default_rng(422)
    x = rng.normal(size=(120, adaptive._WIDTH))
    targets = np.digitize(x[:, 0], (-0.5, 0.4))
    targets[::7] = -1
    model = adaptive.fit(x, targets)
    bundle = tmp_path / "trees.npz"
    model.save(bundle)
    restored = adaptive.DirectionContextModel.load(bundle)
    expected = estimator.predict_proba(x)
    np.testing.assert_allclose(restored.predict(x), expected, atol=1e-14)
    np.testing.assert_allclose(restored.predict(x[:9]), expected[:9], atol=1e-14)
    restored.children[0, 0] = 0
    corrupted = tmp_path / "cycle.npz"
    restored.save(corrupted)
    with pytest.raises(ValueError, match="children"):
        adaptive.DirectionContextModel.load(corrupted)
