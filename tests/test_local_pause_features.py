"""Check that slope contrasts describe one instrument on its original time grid."""

import importlib
from pathlib import Path
from types import ModuleType

import pytest

np = pytest.importorskip("numpy")


@pytest.fixture
def features(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    return importlib.import_module("local_pause_features")


@pytest.mark.parametrize("width", [32, 48])
def test_pause_slowdown_continuation_and_reversal(features: ModuleType, width: int) -> None:
    column = features.CENTER_WIDTHS.index(width) * 2
    start = 64
    center = start + width // 2

    def contrast(middle: float, end: float) -> tuple[float, float]:
        rates = np.full(192, 0.0002)
        rates[start : start + width] *= middle
        rates[start + width :] *= end
        prices = rates.cumsum()
        values = features.local_pause_features(prices)[center, column : column + 2]
        np.testing.assert_allclose(
            features.local_pause_features(1.25 - prices)[center, column : column + 2],
            values,
            atol=1e-12,
        )
        return values[0], values[1]

    agreement, pause = contrast(0, 1)
    assert agreement > 0
    assert pause > contrast(0.5, 1)[1] > 0
    assert contrast(1, 1)[1] == pytest.approx(0, abs=1e-12)
    reversal, reversal_contrast = contrast(0, -1)
    assert reversal == pytest.approx(-agreement)
    assert reversal_contrast < 0


def test_missing_prices_keep_timestamps_and_edges(features: ModuleType) -> None:
    prices = np.arange(128) ** 2 * 1e-6
    prices[[15, 18, 22, 30, 32]] = np.nan
    values = features.local_pause_features(prices)
    assert values.shape == (128, 4)
    assert np.isnan(values[:48, :2]).all() and np.isnan(values[-47:, :2]).all()
    assert np.isnan(values[:56, 2:]).all() and np.isnan(values[-55:, 2:]).all()
    # At position 64 the left, center and right windows start at 16, 48 and 80.
    left = prices[16:48]
    known = np.isfinite(left)
    slope = np.polyfit(np.arange(32)[known], left[known], 1)[0]
    assert values[64, 0] == pytest.approx(np.arcsinh(31 * slope / 0.01))

    prices[16:23] = np.nan
    assert np.isnan(features.local_pause_features(prices)[64, :2]).all()
    constant = features.local_pause_features(np.zeros(112))
    assert np.array_equal(constant[56], np.zeros(4))
    assert np.isnan(features.local_pause_features(np.arange(95))).all()


def test_missing_center_keeps_only_the_observed_flank_agreement(features: ModuleType) -> None:
    prices = np.arange(128) * 0.0002
    prices[48:80] = np.nan
    agreement, flattening = features.local_pause_features(prices)[64, :2]
    assert agreement == pytest.approx(np.arcsinh(31 * 0.0002 / 0.01))
    assert np.isnan(flattening)
