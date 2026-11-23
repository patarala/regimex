from importlib import import_module
from itertools import pairwise
from pathlib import Path
from types import ModuleType

import pytest


@pytest.fixture
def candidate(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    pytest.importorskip("numpy")
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    return import_module("adaptive_piece_evidence")


def test_joint_fit_matches_exhaustive_search_on_a_small_grid(candidate: ModuleType) -> None:
    import numpy as np

    from regimex.directions import AnalysisConfig

    config = AnalysisConfig(min_duration=4)
    x = np.arange(19)
    y = np.r_[np.arange(7) * 0.02, [0.12] * 5, 0.12 - np.arange(7) * 0.018]
    prefix = np.vstack(
        (np.zeros(6), np.column_stack((np.ones(len(x)), x, y, x * x, x * y, y * y)).cumsum(axis=0))
    )
    layouts: list[list[int]] = []

    def extend(cuts: list[int]) -> None:
        layouts.append([*cuts, len(x)])
        for cut in range(cuts[-1] + config.min_duration, len(x) - config.min_duration + 1):
            extend([*cuts, cut])

    extend([0])
    noise = 1e-5

    def cost(cuts: list[int], penalty: float) -> float:
        squared = sum(
            np.square(y[a:b] - np.polyval(np.polyfit(x[a:b], y[a:b], 1), x[a:b])).sum()
            for a, b in pairwise(cuts)
        )
        return float(squared + (len(cuts) - 2) * penalty)

    for result, strength in zip(
        candidate.partitions(prefix, 0, config, noise), (4, 12, 36), strict=True
    ):
        penalty = strength * np.log1p(len(x)) * noise
        assert cost(result, penalty) == pytest.approx(
            min(cost(layout, penalty) for layout in layouts), abs=1e-12
        )


def test_gaps_separate_fitted_evidence_on_the_original_grid(candidate: ModuleType) -> None:
    import numpy as np
    from test_segmentation import experiment

    from regimex.directions import AnalysisConfig

    price = (100 * np.exp(0.001 * np.arange(128))).tolist()
    config = AnalysisConfig()
    full = candidate.features(experiment(price + [None] * 36 + price[::-1]).observations, config)
    changed = candidate.features(
        experiment(price[::-1] + [None] * 36 + price[::-1]).observations, config
    )
    np.testing.assert_array_equal(full[164:], changed[164:])
    assert np.isfinite(full).all() and np.all(full[128:164] == 0)
    assert candidate.features(experiment([1.0, 2.0]).observations, config).shape == (2, 30)
