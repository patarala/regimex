"""Exercise the retrospective execution filter at carried prices and gaps."""

import importlib
from itertools import pairwise
from pathlib import Path

import pytest

from regimex.data_catalog import SnapshotGrid
from regimex.exports import ExportInstrument, InstrumentUnits


def test_execution_filter_uses_trades_and_preserves_missing_prices(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    np = pytest.importorskip("numpy")
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    candidate = importlib.import_module("check_execution_labels")
    observations = importlib.import_module("latent_direction_training").observations
    prices = np.array([3, 3, 9, 9, np.nan, 9, 4, 5, 5], dtype=float)
    counts = np.array([1, 0, 1, 0, -1, 0, 1, 1, 0])
    rows = observations(prices, counts)
    result = candidate.execution_prices(prices, rows)
    np.testing.assert_allclose(result, [3, 3.5, 4, 4.25, np.nan, 4.75, 5, 5, 5])


@pytest.mark.parametrize("method", ["combined", "activity-80"])
def test_candidate_preserves_original_availability_and_interval_statistics(
    monkeypatch: pytest.MonkeyPatch,
    method: str,
) -> None:
    pytest.importorskip("numpy")
    from regimex.directions import observation_blocks, summarize_interval
    from regimex.segmentation import annotate

    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    audit = importlib.import_module("check_execution_label_quality")
    rows, reference = next(
        (rows, reference)
        for shape, view, rows, reference in audit.drift_paths(723, 0, 256)
        if shape == "pause_then_reversal" and view == "masked"
    )
    instrument = ExportInstrument(
        instrument_id="EUR/USD",
        base_asset="EUR",
        quote_asset="USD",
        initial_price=100.0,
        units=InstrumentUnits(
            price_unit="quote_asset_per_base_asset",
            volume_unit="base_asset",
            price_decimal_places=2,
            volume_decimal_places=0,
        ),
    )
    grid = SnapshotGrid(
        experiment_id="executions", step_count=256, instruments=(instrument,), observations=rows
    )
    assert audit.label_sequence(rows, "current") == annotate(grid, audit.PROFILE).intervals
    available = [r for r in reference if r.direction is None]
    candidate = audit.label_sequence(rows, method)
    assert [r for r in candidate if r.direction is None] == available
    assert candidate[0].start == 0 and candidate[-1].stop == len(rows)
    assert all(a.stop == b.start for a, b in pairwise(candidate))
    for interval in candidate:
        assert interval.statistics == summarize_interval(rows, interval.start, interval.stop)
    gap = next(
        (a, b) for a, b, status in observation_blocks(rows, audit.PROFILE) if status != "observed"
    )
    after = tuple(r.model_copy(update={"last_price": 10_000}) for r in rows[gap[1] :])
    changed = audit.label_sequence(rows[: gap[1]] + after, method)
    assert [r for r in changed if r.stop <= gap[0]] == [r for r in candidate if r.stop <= gap[0]]


def test_stress_views_share_latent_pauses_and_full_activity_uses_the_current_rule(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("numpy")
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    audit = importlib.import_module("check_execution_label_quality")
    cases = {
        view: (rows, truth)
        for shape, view, rows, truth in audit.stress_paths(597, 1, crossed=True)
        if shape == "plateau-32"
    }
    original, truth = cases["stress"]
    assert audit.label_sequence(original, "activity-80") == audit.label_sequence(
        original, "current"
    )
    spans = [(r.start, r.stop, r.direction) for r in truth]
    for view in ("dense", "sparse"):
        rows, reference = cases[view]
        assert [(r.start, r.stop, r.direction) for r in reference] == spans
        assert any(r.trade_count == 0 for r in rows)
        assert all(r.last_price == original[r.step].last_price for r in rows if r.trade_count)
