from importlib import import_module
from pathlib import Path
from types import ModuleType

import pytest


@pytest.fixture
def mixed(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    pytest.importorskip("numpy")
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    return import_module("train_mixed_direction")


def test_latent_path_replays_observations_and_separate_drift_targets(mixed: ModuleType) -> None:
    import numpy as np

    from regimex.directions import observation_blocks

    profile = mixed.PROFILE
    seed = mixed.FIRST_SEED
    prices, counts, rows, targets = mixed.generate(seed, 1024, profile)
    again = mixed.generate(seed, 1024, profile)
    for first, second in zip(
        (prices, counts, targets), (again[0], again[1], again[3]), strict=True
    ):
        np.testing.assert_array_equal(first, second)
    assert mixed.observations(prices, counts) == rows
    assert (counts < 0).any()
    assert set(np.unique(targets)).issubset({-1, 0, 1, 2})
    for a, b, status in observation_blocks(rows, profile):
        if status != "observed":
            assert (targets[a:b] == -1).all()
    observation_hash, target_hash = mixed.hashes(prices, counts, targets)
    swapped = np.select([targets == 1, targets == 2], [2, 1], default=targets)
    other_observations, other_targets = mixed.hashes(prices, counts, swapped)
    assert other_observations == observation_hash
    assert other_targets != target_hash


@pytest.mark.parametrize("fraction", [0.1, 0.25, 0.45])
def test_each_source_supplies_its_registered_weight_in_every_class(
    mixed: ModuleType, fraction: float
) -> None:
    import numpy as np

    from regimex.backends.adaptive_forest import FEATURE_NAMES, ForestTrainingRows, training_rows

    def source(lengths: tuple[int, int, int]) -> list[ForestTrainingRows]:
        labels = np.repeat(np.arange(3, dtype=np.int64), lengths)
        x = np.ones((len(labels), len(FEATURE_NAMES)))
        return [training_rows(x, labels)]

    native = source((150, 80, 36))
    latent = source((36, 270, 128))
    total = mixed.class_totals(native, "counts") + mixed.class_totals(latent, "counts")
    balanced = mixed.mixture(native, latent, fraction)
    np.testing.assert_allclose(mixed.class_totals(balanced, "interval_weights"), total)
    np.testing.assert_allclose(
        mixed.class_totals(balanced[1:], "interval_weights"), fraction * total
    )
    np.testing.assert_array_equal(balanced[0].counts, native[0].counts)
