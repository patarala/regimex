"""Exchange accounting, execution priority, participant behavior and label independence."""

import pytest
from pydantic import ValidationError

pytest.importorskip("numpy")

from regimex.market import (
    VIRTUAL_BASE_ASSETS,
    Exchange,
    Wallet,
    market_instruments,
    simulate,
    trend_action,
)
from regimex.market_config import MarketConfig, ParticipantProfile
from regimex.market_sampling import sample_market


def test_price_then_time_priority_partial_fills_and_price_improvement() -> None:
    exchange = Exchange((100,), [Wallet(1000, [10]) for _ in range(4)])
    exchange.submit(0, 0, "sell", 102, 3)
    exchange.submit(1, 0, "sell", 101, 2)
    exchange.submit(2, 0, "sell", 101, 4)
    result = exchange.submit(3, 0, "buy", 105, 4)
    assert [(t.maker, t.price, t.lots) for t in result.trades] == [(1, 101, 2), (2, 101, 2)]
    assert exchange.wallets[3].cash == 596
    assert exchange.wallets[3].reserved_cash == 0
    assert exchange.wallets[3].assets == [14]
    assert exchange.asks[0][0][2].owner == 2
    assert exchange.asks[0][0][2].remaining == 2
    assert exchange.last == [101]
    exchange.check_invariants()


def test_resting_buy_executes_at_its_price_and_releases_only_filled_reservation() -> None:
    exchange = Exchange((100,), [Wallet(1000, [10]), Wallet(1000, [10])])
    exchange.submit(0, 0, "buy", 105, 4)
    result = exchange.submit(1, 0, "sell", 99, 1)
    assert [(t.price, t.lots) for t in result.trades] == [(105, 1)]
    assert exchange.wallets[0].cash == 580
    assert exchange.wallets[0].reserved_cash == 315
    assert exchange.wallets[1].cash == 1105
    exchange.check_invariants()


def test_shared_cash_reservations_prevent_overspending_across_books() -> None:
    exchange = Exchange((100, 100), [Wallet(1000, [2, 2]), Wallet(1000, [2, 2])])
    assert exchange.submit(0, 0, "buy", 100, 8).accepted
    assert not exchange.submit(0, 1, "buy", 100, 3).accepted
    assert exchange.submit(1, 1, "sell", 90, 2).accepted
    assert exchange.history == [[], []]
    assert not exchange.submit(1, 1, "sell", 90, 1).accepted
    assert exchange.last == [100, 100]
    exchange.check_invariants()


def test_self_trade_cancels_incoming_remainder_without_changing_price_or_balances() -> None:
    exchange = Exchange((100,), [Wallet(1000, [10])])
    exchange.submit(0, 0, "sell", 99, 3)
    result = exchange.submit(0, 0, "buy", 101, 5)
    assert result.accepted and result.cancelled_lots == 5 and not result.trades
    assert exchange.wallets[0].cash == 1000
    assert exchange.wallets[0].reserved_cash == 0
    assert exchange.last == [100]
    exchange.check_invariants()


def test_trend_uses_executed_trade_history_and_waits_for_a_full_window() -> None:
    assert trend_action([100, 101], 2, 0.01) is None
    assert trend_action([100, 101, 102], 2, 0.01) == "buy"
    assert trend_action([100, 99, 98], 2, 0.01) == "sell"
    assert trend_action([100, 100, 100], 2, 0.01) is None
    # 203 / 200 - 1 rounds below 0.015 in binary floating point.
    assert trend_action([200, 203], 1, 0.015) == "buy"
    assert trend_action([200, 197], 1, 0.015) == "sell"
    profile = ParticipantProfile(strategy="trend")
    run = simulate(
        MarketConfig(profiles=(profile,), test_participant=profile.model_copy(update={"count": 1})),
        17,
        128,
    )
    assert run.audit.holds == run.audit.events
    assert run.audit.trades == 0
    assert all(row.last_price == 100.0 for row in run.observations)


@pytest.mark.parametrize("side", ["hold", "buy", "sell"])
def test_nontrading_participants_produce_flat_prices_without_artificial_regimes(side: str) -> None:
    profile = ParticipantProfile(
        count=8,
        buy_probability=float(side == "buy"),
        sell_probability=float(side == "sell"),
        hold_probability=float(side == "hold"),
        price_offset_bps=(0.0, 0.0),
        volume_lots=(1, 1),
    )
    market = MarketConfig(
        initial_prices=(100.0, 200.0),
        profiles=(profile,),
        test_participant=profile.model_copy(update={"count": 1}),
    )
    run = simulate(market, 0, 64)
    assert all(row.trade_count == 0 for row in run.observations)
    assert {row.last_price for row in run.observations} == {100.0, 200.0}
    assert all(row.best_ask is None for row in run.observations) if side != "sell" else True
    assert all(row.best_bid is None for row in run.observations) if side != "buy" else True


def test_invalid_strategy_and_market_settings_are_rejected() -> None:
    with pytest.raises(ValidationError, match="sum to one"):
        ParticipantProfile(buy_probability=0.8)
    with pytest.raises(ValidationError, match="volume lots"):
        ParticipantProfile(volume_lots=(10, 1))
    with pytest.raises(ValidationError, match="multiples"):
        MarketConfig(initial_prices=(100.001,))


def test_initial_price_rejects_fractional_ticks_even_when_close_to_a_tick() -> None:
    with pytest.raises(ValidationError, match="multiples"):
        MarketConfig(initial_prices=(0.010000001,), tick_size=0.01)


@pytest.mark.parametrize("price,tick", [(0.3, 0.1), (1234567890.12, 0.01)])
def test_exact_decimal_initial_price_is_preserved_without_trades(price: float, tick: float) -> None:
    hold = ParticipantProfile(
        count=1, buy_probability=0.0, sell_probability=0.0, hold_probability=1.0
    )
    market = MarketConfig(
        initial_prices=(price,), tick_size=tick, profiles=(hold,), test_participant=hold
    )
    run = simulate(market, 0, 2)
    assert run.instruments[0].initial_price == price
    assert run.audit.trades == 0
    assert all(row.last_price == price and row.trade_count == 0 for row in run.observations)


def test_partial_fill_before_self_match_preserves_executions_and_releases_remainder() -> None:
    exchange = Exchange((100,), [Wallet(1000, [10]), Wallet(1000, [10])])
    exchange.submit(1, 0, "sell", 98, 2)
    exchange.submit(0, 0, "sell", 99, 3)
    result = exchange.submit(0, 0, "buy", 101, 5)
    assert [(trade.price, trade.lots) for trade in result.trades] == [(98, 2)]
    assert result.cancelled_lots == 3
    assert exchange.wallets[0].cash == 804
    assert exchange.wallets[0].reserved_cash == 0
    assert exchange.wallets[0].assets == [9]
    assert exchange.wallets[0].reserved_assets == [3]
    exchange.check_invariants()


def test_executions_can_leave_prices_unchanged_at_the_tick_resolution() -> None:
    profile = ParticipantProfile(count=8, price_offset_bps=(-0.1, 0.1))
    market = MarketConfig(
        profiles=(profile,), test_participant=profile.model_copy(update={"count": 1})
    )
    run = simulate(market, 0, 64)
    assert all(row.last_price == 100.0 for row in run.observations)
    assert run.audit.trades > 0


def test_sampled_markets_cover_independent_strategies_and_heterogeneous_profiles() -> None:
    markets = [sample_market(seed, 1 + seed % 2) for seed in range(256)]
    assert {m.test_participant.strategy for m in markets} == {"random", "trend"}
    assert {len(m.profiles) for m in markets} == {1, 2, 3, 4, 5}
    assert {m.profiles[0].strategy for m in markets} == {"random", "trend"}
    assert {len(m.initial_prices) for m in markets} == {1, 2}
    assert len({m.test_participant.volume_lots for m in markets}) > 32
    assert len({m.test_participant.trend_window_trades for m in markets}) > 32
    for seed, market in enumerate(markets):
        assert sample_market(seed, len(market.initial_prices)) == market
        assert all(
            market.test_participant != profile.model_copy(update={"count": 1})
            for profile in market.profiles
        )
        assert sum(p.count for p in market.profiles) <= 160
    with pytest.raises(ValueError, match="one or two"):
        sample_market(0, 3)


def test_journaling_preserves_random_streams_and_records_exact_event_effects() -> None:
    from regimex.market import Balance

    random = ParticipantProfile(count=4, price_offset_bps=(0.0, 0.0), volume_lots=(1, 4))
    trend = ParticipantProfile(count=1, strategy="trend", trend_window_trades=2)
    config = MarketConfig(initial_prices=(100.0, 200.0), profiles=(random,), test_participant=trend)
    ordinary = simulate(config, 41, 32)
    recorded = simulate(config, 41, 32, record_events=True)
    assert ordinary.journal is None
    assert recorded == simulate(config, 41, 32, record_events=True)
    assert ordinary.observations == recorded.observations
    assert ordinary.audit == recorded.audit
    assert recorded.journal is not None
    assert len(recorded.journal) == 160
    assert [event.ordinal for event in recorded.journal] == list(range(160))
    assert {event.status for event in recorded.journal} == {"held", "accepted"}
    profiles = [random] * random.count + [trend]
    exchange = Exchange(
        (10000, 20000),
        [Wallet(config.cash_units(p.cash), [p.asset_lots, p.asset_lots]) for p in profiles],
    )
    traded = 0
    for event in recorded.journal:
        assert event.step == event.ordinal // len(profiles)
        assert event.participant == event.ordinal % len(profiles)
        if event.action != "hold":
            assert event.order_id == exchange.next_order
            assert event.price_ticks is not None and event.lots is not None
            result = exchange.submit(
                event.participant, event.instrument, event.action, event.price_ticks, event.lots
            )
            assert result.accepted == (event.status == "accepted")
            assert result.cancelled_lots == event.cancelled_lots
            assert [
                (t.maker_order_id, t.taker_order_id, t.price, t.lots) for t in result.trades
            ] == [(t.maker_order_id, t.taker_order_id, t.price_ticks, t.lots) for t in event.trades]
        for affected in event.balances_after:
            wallet = exchange.wallets[affected.participant]
            assert affected.balance == Balance(
                cash_units=wallet.cash,
                reserved_cash_units=wallet.reserved_cash,
                asset_lots=tuple(wallet.assets),
                reserved_asset_lots=tuple(wallet.reserved_assets),
            )
        assert exchange.last[event.instrument] == event.last_price_ticks
        traded += len(event.trades)
        exchange.check_invariants()
    assert traded == recorded.audit.trades > 0


def test_journal_distinguishes_rejections_holds_and_empty_books() -> None:
    poor_buyer = ParticipantProfile(
        count=1, buy_probability=1.0, sell_probability=0.0, hold_probability=0.0, cash=0.0
    )
    hold = ParticipantProfile(
        count=1, buy_probability=0.0, sell_probability=0.0, hold_probability=1.0
    )
    config = MarketConfig(profiles=(poor_buyer,), test_participant=hold)
    run = simulate(config, 0, 2, record_events=True)
    assert run.journal is not None
    assert [event.status for event in run.journal] == ["rejected", "held", "rejected", "held"]
    assert [event.order_id for event in run.journal] == [0, None, 1, None]
    assert all(
        event.best_bid_ticks is None and event.best_ask_ticks is None for event in run.journal
    )
    with pytest.raises(ValueError, match="100000 events"):
        simulate(config, 0, 50001, record_events=True)


def test_virtual_asset_names_are_distinct_stable_and_used_in_observations() -> None:
    single = MarketConfig(initial_prices=(100.0,))
    paired = MarketConfig(initial_prices=(100.0, 200.0))
    seen = set()
    for seed in range(64):
        instruments = market_instruments(paired, seed)
        assert instruments == market_instruments(paired, seed)
        assert market_instruments(single, seed) == instruments[:1]
        assert len({instrument.instrument_id for instrument in instruments}) == 2
        for instrument in instruments:
            assert instrument.base_asset in VIRTUAL_BASE_ASSETS
            assert instrument.quote_asset == "USD"
            assert instrument.instrument_id == f"{instrument.base_asset}/USD"
            seen.add(instrument.base_asset)
        run = simulate(paired, seed, 2)
        assert run.instruments == instruments
        assert {row.instrument_id for row in run.observations} == {
            instrument.instrument_id for instrument in instruments
        }
    assert seen == set(VIRTUAL_BASE_ASSETS)


def test_instrument_names_are_independent_of_prices_and_simulation_duration() -> None:
    first = simulate(MarketConfig(initial_prices=(100.0, 200.0)), 41, 2)
    longer = simulate(MarketConfig(initial_prices=(100.0, 200.0)), 41, 4)
    changed = simulate(MarketConfig(initial_prices=(300.0, 400.0)), 41, 2)
    assert first.instruments == longer.instruments
    assert first.observations == longer.observations[: len(first.observations)]
    assert tuple(item.instrument_id for item in first.instruments) == tuple(
        item.instrument_id for item in changed.instruments
    )


@pytest.mark.parametrize("seed", [0, 41, 20261014])
@pytest.mark.parametrize("instrument_count", [1, 2])
def test_sampled_execution_prices_and_balances_remain_consistent(
    seed: int, instrument_count: int
) -> None:
    market = sample_market(seed, instrument_count)
    run = simulate(market, seed, 64)
    assert simulate(market, seed, 64) == run
    assert len(run.observations) == 64 * instrument_count
    previous_prices = {item.instrument_id: item.initial_price for item in run.instruments}
    for observation in run.observations:
        assert observation.last_price is not None
        assert observation.trade_count is not None
        if observation.trade_count == 0:
            assert observation.last_price == previous_prices[observation.instrument_id]
        previous_prices[observation.instrument_id] = observation.last_price
    assert (
        sum(
            balance.cash_units + balance.reserved_cash_units for balance in run.audit.final_balances
        )
        == run.audit.initial_cash_units
    )
    for index, initial_lots in enumerate(run.audit.initial_asset_lots):
        assert (
            sum(
                balance.asset_lots[index] + balance.reserved_asset_lots[index]
                for balance in run.audit.final_balances
            )
            == initial_lots
        )
