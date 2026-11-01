"""Deterministic participant events and a price/time priority limit-order exchange."""

import hashlib
import heapq
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Literal

import numpy as np
from pydantic import Field

from regimex.contracts import Count, Instrument, Observation, StrictModel
from regimex.market_config import MarketConfig, ParticipantProfile

Side = Literal["buy", "sell"]

VIRTUAL_BASE_ASSETS = ("BTC", "ETH", "SOL", "XRP", "ADA", "LTC", "DOT", "DOGE")


def market_instruments(config: MarketConfig, seed: int) -> tuple[Instrument, ...]:
    """Name virtual assets without consuming participant random streams.

    Names identify synthetic assets; their prices and participant behavior are
    not calibrated to the real currencies. A seed keeps its names when labels,
    observation duration, or the number of instruments change.
    """
    assets = sorted(
        VIRTUAL_BASE_ASSETS,
        key=lambda asset: hashlib.sha256(f"market-assets/1.0\0{seed}\0{asset}".encode()).digest(),
    )
    return tuple(
        Instrument(
            instrument_id=f"{asset}/USD",
            base_asset=asset,
            quote_asset="USD",
            initial_price=price,
        )
        for asset, price in zip(
            assets[: len(config.initial_prices)], config.initial_prices, strict=True
        )
    )


@dataclass
class Wallet:
    cash: int
    assets: list[int]
    reserved_cash: int = 0
    reserved_assets: list[int] = field(default_factory=list)


@dataclass
class Order:
    identifier: int
    owner: int
    instrument: int
    side: Side
    price: int
    remaining: int


@dataclass(frozen=True)
class Trade:
    maker: int
    taker: int
    instrument: int
    price: int
    lots: int
    maker_order_id: int
    taker_order_id: int


@dataclass(frozen=True)
class Submission:
    accepted: bool
    trades: tuple[Trade, ...] = ()
    cancelled_lots: int = 0


class Exchange:
    """All balances use integer tick-lot units; prices and sizes use ticks and lots."""

    def __init__(self, initial_prices: tuple[int, ...], wallets: list[Wallet]) -> None:
        if not initial_prices or any(price <= 0 for price in initial_prices):
            raise ValueError("initial prices must be positive")
        self.wallets = wallets
        self.last = list(initial_prices)
        self.history: list[list[int]] = [[] for _ in initial_prices]
        self.bids: list[list[tuple[int, int, Order]]] = [[] for _ in initial_prices]
        self.asks: list[list[tuple[int, int, Order]]] = [[] for _ in initial_prices]
        self.next_order = 0
        for wallet in wallets:
            if (
                wallet.cash < 0
                or len(wallet.assets) != len(initial_prices)
                or any(value < 0 for value in wallet.assets)
                or wallet.reserved_cash
                or any(wallet.reserved_assets)
            ):
                raise ValueError("initial wallets must contain nonnegative unreserved balances")
            wallet.reserved_assets = [0] * len(initial_prices)
        self.initial_cash = sum(wallet.cash for wallet in wallets)
        self.initial_assets = [
            sum(w.assets[i] for w in wallets) for i in range(len(initial_prices))
        ]

    def _release(self, order: Order) -> None:
        wallet = self.wallets[order.owner]
        if order.side == "buy":
            amount = order.price * order.remaining
            wallet.reserved_cash -= amount
            wallet.cash += amount
        else:
            wallet.reserved_assets[order.instrument] -= order.remaining
            wallet.assets[order.instrument] += order.remaining
        order.remaining = 0

    def submit(self, owner: int, instrument: int, side: Side, price: int, lots: int) -> Submission:
        if (
            not 0 <= owner < len(self.wallets)
            or not 0 <= instrument < len(self.last)
            or side not in ("buy", "sell")
            or price <= 0
            or lots <= 0
        ):
            raise ValueError("invalid order")
        order = Order(self.next_order, owner, instrument, side, price, lots)
        self.next_order += 1
        wallet = self.wallets[owner]
        if side == "buy":
            if wallet.cash < price * lots:
                return Submission(False)
            wallet.cash -= price * lots
            wallet.reserved_cash += price * lots
        else:
            if wallet.assets[instrument] < lots:
                return Submission(False)
            wallet.assets[instrument] -= lots
            wallet.reserved_assets[instrument] += lots
        opposite = self.asks[instrument] if side == "buy" else self.bids[instrument]
        trades: list[Trade] = []
        while opposite and order.remaining:
            maker = opposite[0][2]
            if (side == "buy" and price < maker.price) or (side == "sell" and price > maker.price):
                break
            if maker.owner == owner:
                # Cancel the incoming remainder: no self-trade or crossed resting book.
                cancelled = order.remaining
                self._release(order)
                return Submission(True, tuple(trades), cancelled)
            quantity = min(order.remaining, maker.remaining)
            buyer, seller = (order, maker) if side == "buy" else (maker, order)
            buyer_wallet, seller_wallet = self.wallets[buyer.owner], self.wallets[seller.owner]
            buyer_wallet.reserved_cash -= buyer.price * quantity
            buyer_wallet.cash += (buyer.price - maker.price) * quantity
            buyer_wallet.assets[instrument] += quantity
            seller_wallet.reserved_assets[instrument] -= quantity
            seller_wallet.cash += maker.price * quantity
            maker.remaining -= quantity
            order.remaining -= quantity
            self.last[instrument] = maker.price
            self.history[instrument].append(maker.price)
            trades.append(
                Trade(
                    maker.owner,
                    owner,
                    instrument,
                    maker.price,
                    quantity,
                    maker.identifier,
                    order.identifier,
                )
            )
            if maker.remaining == 0:
                heapq.heappop(opposite)
        if order.remaining:
            book = self.bids[instrument] if side == "buy" else self.asks[instrument]
            heapq.heappush(book, (-price if side == "buy" else price, order.identifier, order))
        return Submission(True, tuple(trades))

    def check_invariants(self) -> None:
        reserved_cash = [0] * len(self.wallets)
        reserved_assets = [[0] * len(self.last) for _ in self.wallets]
        for instrument, (bids, asks) in enumerate(zip(self.bids, self.asks, strict=True)):
            if bids and asks and bids[0][2].price >= asks[0][2].price:
                raise ValueError("resting order book is crossed")
            for _, _, order in bids:
                if order.remaining <= 0:
                    raise ValueError("completed order remains in the book")
                reserved_cash[order.owner] += order.price * order.remaining
            for _, _, order in asks:
                if order.remaining <= 0:
                    raise ValueError("completed order remains in the book")
                reserved_assets[order.owner][instrument] += order.remaining
        for owner, wallet in enumerate(self.wallets):
            if (
                min(wallet.cash, wallet.reserved_cash, *wallet.assets, *wallet.reserved_assets) < 0
                or wallet.reserved_cash != reserved_cash[owner]
                or wallet.reserved_assets != reserved_assets[owner]
            ):
                raise ValueError("balances disagree with active order reservations")
        if sum(w.cash + w.reserved_cash for w in self.wallets) != self.initial_cash:
            raise ValueError("exchange did not conserve quote assets")
        for instrument, initial in enumerate(self.initial_assets):
            if (
                sum(w.assets[instrument] + w.reserved_assets[instrument] for w in self.wallets)
                != initial
            ):
                raise ValueError("exchange did not conserve base assets")


def trend_action(history: list[int], window: int, threshold: float) -> Side | None:
    if len(history) <= window:
        return None
    reference = history[-1 - window]
    threshold_ratio = Fraction(str(threshold))
    change = (history[-1] - reference) * threshold_ratio.denominator
    boundary = threshold_ratio.numerator * reference
    if change >= boundary:
        return "buy"
    if change <= -boundary:
        return "sell"
    return None


class Balance(StrictModel):
    cash_units: int = Field(ge=0)
    reserved_cash_units: int = Field(ge=0)
    asset_lots: tuple[int, ...]
    reserved_asset_lots: tuple[int, ...]


class MarketAudit(StrictModel):
    events: Count
    holds: Count
    accepted_orders: Count
    rejected_orders: Count
    self_trade_cancellations: Count
    trades: Count
    initial_cash_units: int = Field(ge=0)
    initial_asset_lots: tuple[int, ...]
    final_balances: tuple[Balance, ...]


class JournalTrade(StrictModel):
    maker: Count
    taker: Count
    maker_order_id: Count
    taker_order_id: Count
    price_ticks: int = Field(gt=0)
    lots: int = Field(gt=0)


class JournalBalance(StrictModel):
    participant: Count
    balance: Balance


class MarketEvent(StrictModel):
    """An ordered event and its effects, in exact integer settlement units."""

    ordinal: Count
    step: Count
    participant: Count
    instrument: Count
    action: Literal["buy", "sell", "hold"]
    status: Literal["held", "accepted", "rejected"]
    order_id: Count | None
    price_ticks: int | None
    lots: int | None
    cancelled_lots: Count = 0
    trades: tuple[JournalTrade, ...] = ()
    balances_after: tuple[JournalBalance, ...]
    last_price_ticks: int = Field(gt=0)
    best_bid_ticks: int | None
    best_ask_ticks: int | None


@dataclass(frozen=True)
class MarketRun:
    instruments: tuple[Instrument, ...]
    observations: tuple[Observation, ...]
    audit: MarketAudit
    journal: tuple[MarketEvent, ...] | None = None


def _balance(wallet: Wallet) -> Balance:
    return Balance(
        cash_units=wallet.cash,
        reserved_cash_units=wallet.reserved_cash,
        asset_lots=tuple(wallet.assets),
        reserved_asset_lots=tuple(wallet.reserved_assets),
    )


def _record_event(
    exchange: Exchange,
    step: int,
    owner: int,
    instrument: int,
    side: Side | None,
    order_id: int | None = None,
    price: int | None = None,
    lots: int | None = None,
    submission: Submission | None = None,
) -> MarketEvent:
    trades = submission.trades if submission else ()
    affected = sorted({owner, *(trade.maker for trade in trades)})
    return MarketEvent(
        ordinal=step * len(exchange.wallets) + owner,
        step=step,
        participant=owner,
        instrument=instrument,
        action=side or "hold",
        status="held" if submission is None else "accepted" if submission.accepted else "rejected",
        order_id=order_id,
        price_ticks=price,
        lots=lots,
        cancelled_lots=submission.cancelled_lots if submission else 0,
        trades=tuple(
            JournalTrade(
                maker=t.maker,
                taker=t.taker,
                maker_order_id=t.maker_order_id,
                taker_order_id=t.taker_order_id,
                price_ticks=t.price,
                lots=t.lots,
            )
            for t in trades
        ),
        balances_after=tuple(
            JournalBalance(participant=i, balance=_balance(exchange.wallets[i])) for i in affected
        ),
        last_price_ticks=exchange.last[instrument],
        best_bid_ticks=exchange.bids[instrument][0][2].price if exchange.bids[instrument] else None,
        best_ask_ticks=exchange.asks[instrument][0][2].price if exchange.asks[instrument] else None,
    )


def simulate(
    config: MarketConfig, seed: int, steps: int, *, record_events: bool = False
) -> MarketRun:
    profiles = [profile for profile in config.profiles for _ in range(profile.count)]
    profiles.append(config.test_participant)
    if steps < 1:
        raise ValueError("simulation requires at least one step")
    if record_events and steps * len(profiles) > 100_000:
        raise ValueError("event journaling supports at most 100000 events per experiment")
    instruments = market_instruments(config, seed)
    exchange = Exchange(
        tuple(config.price_ticks(price) for price in config.initial_prices),
        [
            Wallet(
                config.cash_units(profile.cash),
                [profile.asset_lots] * len(instruments),
            )
            for profile in profiles
        ],
    )
    rngs = [
        np.random.Generator(np.random.PCG64(np.random.SeedSequence([seed, i])))
        for i in range(len(profiles))
    ]
    observations: list[Observation] = []
    journal: list[MarketEvent] | None = [] if record_events else None
    price_unit = Fraction(str(config.tick_size))
    volume_unit = Fraction(str(config.lot_size))
    holds = accepted = rejected = cancellations = trades = 0
    for step in range(steps):
        counts = [[0, 0, 0] for _ in instruments]
        volumes = [[0, 0, 0] for _ in instruments]
        # All events have interval one; ties use participant ordinal (event insertion order).
        for owner, (profile, rng) in enumerate(zip(profiles, rngs, strict=True)):
            instrument = int(rng.integers(len(instruments)))
            side = _action(profile, exchange.history[instrument], rng)
            if side is None:
                holds += 1
                if journal is not None:
                    journal.append(_record_event(exchange, step, owner, instrument, side))
                continue
            price = max(
                1,
                round(
                    exchange.last[instrument] * (1 + rng.uniform(*profile.price_offset_bps) / 10000)
                ),
            )
            if price > 10**12:
                raise ValueError("simulated price exceeds supported tick range")
            lots = int(rng.integers(profile.volume_lots[0], profile.volume_lots[1] + 1))
            order_id = exchange.next_order
            result = exchange.submit(owner, instrument, side, price, lots)
            if journal is not None:
                journal.append(
                    _record_event(
                        exchange, step, owner, instrument, side, order_id, price, lots, result
                    )
                )
            if not result.accepted:
                rejected += 1
                continue
            accepted += 1
            column = 1 if side == "buy" else 2
            counts[instrument][column] += 1
            volumes[instrument][column] += lots
            cancellations += result.cancelled_lots > 0
            trades += len(result.trades)
            counts[instrument][0] += len(result.trades)
            volumes[instrument][0] += sum(trade.lots for trade in result.trades)
        for i, instrument_info in enumerate(instruments):
            observations.append(
                Observation(
                    instrument_id=instrument_info.instrument_id,
                    step=step,
                    last_price=float(exchange.last[i] * price_unit),
                    trade_count=counts[i][0],
                    trade_volume=float(volumes[i][0] * volume_unit),
                    buy_order_count=counts[i][1],
                    buy_order_volume=float(volumes[i][1] * volume_unit),
                    sell_order_count=counts[i][2],
                    sell_order_volume=float(volumes[i][2] * volume_unit),
                    best_bid=float(exchange.bids[i][0][2].price * price_unit)
                    if exchange.bids[i]
                    else None,
                    best_ask=float(exchange.asks[i][0][2].price * price_unit)
                    if exchange.asks[i]
                    else None,
                )
            )
    exchange.check_invariants()
    audit = MarketAudit(
        events=steps * len(profiles),
        holds=holds,
        accepted_orders=accepted,
        rejected_orders=rejected,
        self_trade_cancellations=cancellations,
        trades=trades,
        initial_cash_units=exchange.initial_cash,
        initial_asset_lots=tuple(exchange.initial_assets),
        final_balances=tuple(_balance(w) for w in exchange.wallets),
    )
    return MarketRun(
        instruments, tuple(observations), audit, tuple(journal) if journal is not None else None
    )


def _action(
    profile: ParticipantProfile, history: list[int], rng: np.random.Generator
) -> Side | None:
    if profile.strategy == "trend":
        return trend_action(history, profile.trend_window_trades, profile.trend_threshold)
    draw = rng.random()
    if draw < profile.buy_probability:
        return "buy"
    if draw < profile.buy_probability + profile.sell_probability:
        return "sell"
    return None
