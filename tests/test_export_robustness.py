"""Paired export masks must preserve the source and distinguish zeros from unavailable data."""

import pytest

np = pytest.importorskip("numpy")
robustness = pytest.importorskip("regimex.export_robustness")


def test_masks_preserve_grid_and_leave_unmasked_fields_alone() -> None:
    from regimex.data_catalog import SourceSnapshot, create_snapshot_member
    from regimex.market_export import sample_snapshot
    from regimex.market_identity import market_seed_group

    pristine = sample_snapshot(seed=94001, steps=300, instruments=2)
    group = market_seed_group(94001)
    start, stop = robustness.WINDOW
    for scenario in robustness.SCENARIOS:
        masked = robustness.masked_export(pristine, scenario)
        assert masked.time_grid == pristine.time_grid and masked.instruments == pristine.instruments
        assert masked.source.run_id == pristine.source.run_id
        assert masked.content_sha256 != pristine.content_sha256
        source = SourceSnapshot(source_namespace="paired", split_group=group, export=masked)
        assert create_snapshot_member(source, "test").split_group == group
        for before, after in zip(pristine.observations, masked.observations, strict=True):
            assert (before.step, before.instrument_id) == (after.step, after.instrument_id)
            if scenario == "price_only":
                assert (before.last_price, before.price_state) == (
                    after.last_price,
                    after.price_state,
                )
                assert (after.trade_volume, after.trade_count, after.buy_order_count) == (None,) * 3
                assert (after.bid_state, after.ask_state) == ("unknown", "unknown")
            elif start <= after.step < stop:
                assert after.last_price is None and after.price_state == "unknown"
                expected = (
                    (before.trade_count, before.trade_volume)
                    if scenario == "price_gap"
                    else (None, None)
                )
                assert (after.trade_count, after.trade_volume) == expected
                assert after.best_bid == before.best_bid and after.best_ask == before.best_ask
            else:
                assert before == after


def test_flips_and_f1_use_identical_eligible_points_and_count_lost_coverage() -> None:
    no_trades, missing = robustness.NO_TRADES, robustness.INSUFFICIENT
    reference = np.array([0, 1, 2, no_trades, missing])
    pristine = np.array([0, 1, 2, no_trades, missing])
    masked = np.array([1, missing, 2, missing, 0])
    window = np.array([False, True, False, True, False])
    zero = np.array([False, False, False, True, False])

    result = robustness.paired_metrics(reference, pristine, masked, window, zero)
    assert result["common_points"] == 2
    assert result["changed_direction"] == 1
    assert result["lost_coverage"] == result["gained_coverage"] == 1
    assert result["pristine_macro_f1"] == pytest.approx(2 / 3)
    assert result["masked_macro_f1"] == pytest.approx(1 / 3)
    group = result["groups"]["window_original_zero"]
    assert group["pristine"] == {"observed": 0, "no_trades": 1, "insufficient_data": 0}
    assert group["masked"] == {"observed": 0, "no_trades": 0, "insufficient_data": 1}
    unavailable = np.full(5, missing)
    empty = robustness.paired_metrics(reference, pristine, unavailable, window, zero)
    assert empty["common_points"] == 0
    assert empty["pristine_macro_f1"] is None and empty["masked_macro_f1"] is None
