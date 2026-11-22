"""The altered price-path challenge keeps execution schedules paired and weights frozen."""

import json
from importlib import import_module
from pathlib import Path

import pytest


def test_replay_checks_transfer_reservation_before_loading_model(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    pytest.importorskip("torch")
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    transfer = import_module("check_pause_recovery_transfer")
    comparison = import_module("compare_pause_recovery")
    monkeypatch.setattr(comparison, "read_freeze", lambda _: {"winner": comparison.CONTROL})
    (tmp_path / "plan.json").write_text(json.dumps(transfer.plan()))
    other = tmp_path / "other"
    other.mkdir()
    (other / "plan.json").write_text(json.dumps({"protocol": {"cohorts": {"transfer": [0]}}}))
    with pytest.raises(ValueError, match="different transfer groups"):
        transfer.score(tmp_path, other, tmp_path / "model")
    assert not (tmp_path / "opened.json").exists()


@pytest.mark.parametrize("index", (0, 1))
def test_transfer_shapes_share_executions_and_separate_drift(
    monkeypatch: pytest.MonkeyPatch, index: int
) -> None:
    pytest.importorskip("torch")
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    transfer = import_module("check_pause_recovery_transfer")
    # Exercise the generator on a fixture seed outside the reserved cohorts.
    panel = list(transfer.paths(1230 + 2 * index, index))
    inputs = transfer.observed_inputs([case[-1] for case in panel])
    assert inputs.row_count == 8 * transfer.STEPS
    for market in transfer.MARKETS:
        cases = {
            shape: (expected, rows) for name, shape, expected, _, rows in panel if name == market
        }
        pauses, reference = cases["pause"]
        assert len(pauses) == 2 and all(32 <= stop - start <= 63 for start, stop in pauses)
        for shape, (expected, rows) in cases.items():
            assert [r.trade_count for r in rows] == [r.trade_count for r in reference]
            if shape != "pause":
                assert not expected
        end_prices = [
            cases[shape][1][-1].last_price for shape in ("pullback", "pause", "slowdown", "steady")
        ]
        assert end_prices == sorted(end_prices, reverse=bool(index)) and len(set(end_prices)) == 4
