"""Known-direction replay must score the designed drift and compare the same observed steps."""

from importlib import import_module
from pathlib import Path
from types import ModuleType

import pytest


@pytest.fixture
def transfer(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    pytest.importorskip("torch")
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    return import_module("check_direction_context_transfer")


@pytest.mark.parametrize("index", (0, 3))
def test_direction_replay_uses_scheduled_truth_and_preserves_available_pairs(
    transfer: ModuleType, index: int
) -> None:
    import numpy as np

    cases = list(transfer.paths(4321, index))
    assert transfer.inputs_for(cases).row_count == 20 * transfer.STEPS
    trend = "decline" if index == 0 else "growth"
    opposite = "growth" if index == 0 else "decline"
    directions = {
        "steady": [trend],
        "level-shift": ["sideways"],
        "plateau": [trend, "sideways", trend],
        "reversal": [trend, opposite],
        "pause-before-reversal": [trend, "sideways", opposite],
    }
    for complete, masked in zip(cases[::2], cases[1::2], strict=True):
        assert (complete.view, masked.view) == ("complete", "masked")
        assert (complete.shape, complete.market) == (masked.shape, masked.market)
        labels = complete.directions
        starts = np.r_[0, np.flatnonzero(np.diff(labels)) + 1]
        assert [transfer.DIRECTION_NAMES[labels[i]] for i in starts] == directions[complete.shape]
        full, partial = (transfer.references(case) for case in (complete, masked))
        assert [(r.start, r.stop, r.status) for r in full if r.direction is None] == [
            (*transfer.NO_TRADES, "no_trades")
        ]
        assert [(r.start, r.stop, r.status) for r in partial if r.direction is None] == [
            (*transfer.NO_TRADES, "no_trades"),
            (*transfer.MISSING, "insufficient_data"),
        ]
        for reference in (full, partial):
            for interval in reference:
                if interval.direction is not None:
                    assert interval.stop - interval.start >= transfer.PROFILE.min_duration
                    assert np.all(
                        labels[interval.start : interval.stop]
                        == transfer.DIRECTION_NAMES.index(interval.direction)
                    )
        for i, (before, after) in enumerate(zip(complete.rows, masked.rows, strict=True)):
            if transfer.MISSING[0] <= i < transfer.MISSING[1]:
                assert after.last_price is None and after.trade_count is None
            else:
                assert after == before
        # Price endpoints witness both signs even when trading is sparse.
        if complete.shape == "reversal":
            active_prices = [
                r.last_price for r in complete.rows if r.step >= starts[1] and r.trade_count
            ]
            assert (active_prices[-1] > active_prices[0]) == (opposite == "growth")


def test_transfer_gate_checks_minorities_even_when_pooled_score_rises(transfer: ModuleType) -> None:
    def metrics(macro: float, sideways: float) -> dict[str, object]:
        return {
            "macro_f1": macro,
            "per_class": {
                k: {"f1": sideways if k == "sideways" else 0.75} for k in transfer.DIRECTION_NAMES
            },
        }

    control = metrics(0.7, 0.6)
    exact = transfer.gate(control, metrics(0.67, 0.54))
    assert exact["pooled"] and all(exact["per_class"].values())
    regression = transfer.gate(control, metrics(0.72, 0.53))
    assert regression["pooled"] and not regression["per_class"]["sideways"]


def test_transfer_reservation_rejects_exposure_and_late_selection(
    transfer: ModuleType, tmp_path: Path
) -> None:
    prior = tmp_path / "prior.json"
    prior.write_text(f'{{"excluded_seeds": [{transfer.GROUPS[0]}]}}')
    with pytest.raises(ValueError, match="previously used seeds"):
        transfer.reserve(tmp_path / "stress", tmp_path, prior)
    (tmp_path / "freeze.json").touch()
    with pytest.raises(ValueError, match="before freezing"):
        transfer.reserve(tmp_path / "stress", tmp_path, prior)
    assert not (tmp_path / "stress" / "plan.json").exists()
