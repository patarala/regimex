from importlib import import_module
from pathlib import Path
from types import ModuleType

import pytest


@pytest.fixture
def training(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    pytest.importorskip("numpy")
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    return import_module("balanced_forest_training")


def test_class_counts_are_preserved_while_shorter_intervals_receive_more_examples(
    training: ModuleType,
) -> None:
    import numpy as np

    labels = np.repeat([0, 1, 0, -1, 2], [32, 100, 512, 40, 100]).astype(np.int64)
    weights = training.interval_weights(labels)
    assert weights[0] == 2 and weights[140] == 0.5
    assert np.all(weights[labels == -1] == 0)
    rows = training._sample(labels, weights)
    np.testing.assert_array_equal(np.bincount(labels[rows]), np.bincount(labels[labels >= 0]))
    assert np.count_nonzero(rows < 32) > 32
    np.testing.assert_array_equal(training._sample(labels, weights), rows)
    weights[0] = 0
    with pytest.raises(ValueError, match="positive"):
        training._sample(labels, weights)


def test_price_reflection_exchanges_direction_and_preserves_shape(training: ModuleType) -> None:
    import numpy as np
    from test_segmentation import experiment

    from regimex.backends.adaptive_forest import features
    from regimex.directions import AnalysisConfig

    x = np.arange(32)
    price = 100 * np.exp(x * 0.001 + 0.002 * np.sin(x / 3))
    original = features(experiment(price.tolist()).observations, AnalysisConfig())
    reflected = features(experiment((10000 / price).tolist()).observations, AnalysisConfig())
    original[:, training._SIGNED] *= -1
    np.testing.assert_allclose(original, reflected, atol=1e-10)
    assert training._REFLECTED_CLASSES.tolist() == [0, 2, 1]
